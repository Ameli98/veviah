// Timing check for the resident solver (2026-11-03): perform calls with a hand on a leg on a skirted model (cloth path).
// Run: HSIN_VRM_MODEL=/abs/path/model.vrm node test/resident.mjs
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const client = new Client({ name: 'hsin-resident', version: '0' });
await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(HERE, '../server.mjs')],
  env: { HSIN_COMPANION_PORT: '8799', ...process.env }, stderr: 'inherit' }));   // 8799: never the running app's companion (set HSIN_COMPANION_PORT to aim at one)
const sit = { say: 'Hmm.', emotion: 'neutral', posture: 'side_left',
  keyframes: [{ t: 0 }, { t: 1.5, right_arm: { touch: 'thigh' }, left_arm: { touch: 'floor' } }, { t: 2.2 }] };
const stand = { say: 'Hi.', emotion: 'happy', posture: 'stand',
  keyframes: [{ t: 0 }, { t: 1.0, right_arm: { hand: 'head out beside' } }, { t: 1.6 }] };
for (const [name, a] of [['sit+thigh #1', sit], ['stand', stand], ['sit+thigh #2', sit], ['sit+thigh #3', sit]]) {
  const t0 = Date.now();
  const r = await client.callTool({ name: 'perform', arguments: a });
  console.log(`${name}: ${Date.now() - t0} ms ->`, r.content?.[0]?.text?.slice(0, 80));
  await new Promise((res) => setTimeout(res, name === 'sit+thigh #1' ? 8000 : 200));   // let the warm-up finish once
}
await client.close();
