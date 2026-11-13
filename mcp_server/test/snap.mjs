// Window snapping, server side (2026-11-11): the companion's /rest gives the ledge rest and the leaning-on-a-wall rest,
// /state keeps where she is, /idle and /touch take wall. Run: node test/snap.mjs [model]
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = '8797', base = `http://127.0.0.1:${PORT}`;
const model = process.argv[2] ?? 'test1';
let bad = 0;
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) bad++; };
const env = { ...process.env, HSIN_COMPANION_PORT: PORT, HSIN_VRM_MODEL: path.join(HERE, `../../vrm_player/public/models/${model}.vrm`) };
const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env, stdio: ['ignore', 'inherit', 'pipe'] });
let err = '';
comp.stderr.on('data', (d) => { err += d; });
await new Promise((resolve) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) resolve(); }));
const get = async (u) => (await fetch(base + u)).json();
let t0 = Date.now();
const ledge = await get('/rest?posture=ledge');
check(ledge.length === 1, `ledge rest solved in ${Date.now() - t0} ms: ${JSON.stringify(ledge).slice(0, 120)}`);
if (ledge[0]) { const m = await get(`/motion/${ledge[0].id}`); check(m.root && m.root[0][1] < -0.3, `ledge rest sits (root y ${m.root?.[0]?.[1]})`); }
t0 = Date.now();
const wall = await get('/rest?posture=stand&wall=left');
check(wall.length === 1 && wall[0].id.includes('wallleft'), `wall rest solved in ${Date.now() - t0} ms: ${wall[0]?.id}`);
await fetch(base + '/state', { method: 'POST', body: JSON.stringify({ posture: 'ledge', ledge: true }) });
const st = await get('/state');
check(st.posture === 'ledge' && st.ledge === true && st.wall === null, `state: ${JSON.stringify(st)}`);
t0 = Date.now();
const idle = await get('/idle?posture=stand&wall=right');
check(idle.length >= 3, `wall idle list (${idle.length}) in ${Date.now() - t0} ms`);
const lt = await get('/touch?posture=ledge');
check(lt.length >= 8, `ledge touch list (${lt.length})`);
comp.kill();
if (bad) console.log(err.split('\n').filter((l) => /skipped|error/i.test(l)).slice(-12).join('\n'));
console.log(bad ? `${bad} FAILED` : 'all ok'); process.exit(bad ? 1 : 0);
