// The user's VRM and the solver's files for it, shared by the MCP server (server.mjs) and the desktop companion
// (companion.mjs). Env: HSIN_VRM_MODEL = path to the .vrm (default: the default sample model).
import { createHash } from 'node:crypto';
import { existsSync, readdirSync } from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// the default model: the repo's sample, or a release's renamed one ('VRM Sample.vrm', 2026-11-13: the MCP server read
// the old name at start and crashed, Claude Desktop showed 'Server disconnected'), else any bundled .vrm
const MODELS_DIR = path.join(ROOT, 'vrm_player/public/models');
function defaultModel() {
  for (const n of ['VRM1_Constraint_Twist_Sample.vrm', 'VRM Sample.vrm']) if (existsSync(path.join(MODELS_DIR, n))) return path.join(MODELS_DIR, n);
  try { const any = readdirSync(MODELS_DIR).filter((f) => f.endsWith('.vrm')).sort()[0]; if (any) return path.join(MODELS_DIR, any); } catch { /* none */ }
  return path.join(MODELS_DIR, 'VRM1_Constraint_Twist_Sample.vrm');
}
const ENV_MODEL = process.env.HSIN_VRM_MODEL && existsSync(process.env.HSIN_VRM_MODEL) ? process.env.HSIN_VRM_MODEL : null;
export const MODEL_PATH = ENV_MODEL || defaultModel();
export const MODEL_NAME = path.basename(MODEL_PATH, '.vrm');
const SKELETONS = path.join(ROOT, 'motion_loop/motion_loop/skeletons');
const FITS = path.join(ROOT, 'motion_loop/motion_loop/body_fits');
// A model counts as measured when its skeleton and body fit exist AND were measured from this very file
// (skeletons/<name>.sha256): a model re-exported under the same name is measured again.
// modelInfo(path) works for any .vrm (2026-11-13, Q13a: the MCP server asks the app which model is in use, so the
// AI program's MCP config holds no model path); measured() / saveFit() without a model mean this process's MODEL_PATH.
const infos = new Map();
export async function modelInfo(p = MODEL_PATH) {
  if (!infos.has(p)) {
    const name = path.basename(p, '.vrm');
    infos.set(p, { path: p, name, hash: createHash('sha256').update(await fs.readFile(p)).digest('hex').slice(0, 16) });
  }
  return infos.get(p);
}
export const MODEL_HASH = (await modelInfo().catch(() => ({ hash: null }))).hash;   // a missing file must not stop the server
export async function measured(m) {
  m ??= await modelInfo();
  try {
    await fs.access(path.join(SKELETONS, `${m.name}.json`)); await fs.access(path.join(FITS, `${m.name}.json`));
    return (await fs.readFile(path.join(SKELETONS, `${m.name}.sha256`), 'utf-8')).trim() === m.hash;
  } catch { return false; }
}

// A new model's measurements from the player (vrm_player/src/measure.js). Returns an error text or null.
export async function saveFit(skeleton, fit, m) {
  m ??= await modelInfo();
  const need = ['hips', 'spine', 'head', 'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'rightUpperLeg',
    'rightLowerLeg', 'rightFoot', 'leftUpperArm', 'leftLowerArm', 'leftHand', 'rightUpperArm', 'rightLowerArm', 'rightHand'];
  const lacking = need.filter((b) => !skeleton?.[b]);
  if (lacking.length || !fit?.segments) return `model lacks ${lacking.join(', ') || 'a body fit'}`;
  await fs.writeFile(path.join(SKELETONS, `${m.name}.json`), JSON.stringify(skeleton, null, 1));
  await fs.writeFile(path.join(FITS, `${m.name}.json`), JSON.stringify(fit));
  await fs.writeFile(path.join(SKELETONS, `${m.name}.sha256`), m.hash + '\n');
  return null;
}

// Solved motions are files (Claude Desktop may run a tool and read its resources in different processes).
// Where things live (2026-11-12, step 4: the backend runs on Windows too). WSL / Linux keep ~/.cache/hsin and
// ~/.config/hsin; Windows uses %LOCALAPPDATA%\\veviah\\{cache,config} (env HSIN_DATA overrides the base on both).
const WIN = process.platform === 'win32';
const DATA = process.env.HSIN_DATA || (WIN ? path.join(process.env.LOCALAPPDATA || os.homedir(), 'veviah') : null);
export const CACHE_DIR = DATA ? path.join(DATA, 'cache') : path.join(os.homedir(), '.cache/hsin');
export const CONFIG_DIR = DATA ? path.join(DATA, 'config') : path.join(os.homedir(), '.config/hsin');
export const STORE = path.join(CACHE_DIR, 'motions');
// the solver's python: the embedded Windows python of the app (backend/python), or the repo's venv in WSL
// the solver's environment: UTF-8 for stdin / stdout / files whatever the Windows locale (paths may be in any language)
export const PY_ENV = { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' };
export const PYTHON = process.env.HSIN_PYTHON || (WIN ? path.join(ROOT, 'python', 'python.exe') : path.join(ROOT, '.venv/bin/python'));
export const motionFile = (id) => path.join(STORE, `${String(id).replace(/[^\w-]/g, '')}.json`);

// Where the desktop companion listens (companion.mjs); the MCP server posts every solved motion there.
export const COMPANION_PORT = Number(process.env.HSIN_COMPANION_PORT || 8790);
