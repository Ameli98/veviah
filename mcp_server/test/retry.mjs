// perform results (2026-11-11): the posture is kept when `posture` is left out; a motion the rule filter rejects is not
// shown and goes back once; a second rejection in a row is replaced by a safe version (arms at rest; a long skirt's
// impossible posture becomes a side sit) and shown; results are internal notes. Run: node test/retry.mjs
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';
const HERE = path.dirname(fileURLToPath(import.meta.url));
let bad = 0;
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) bad++; };
const client = new Client({ name: 'hsin-retry', version: '0' });
await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(HERE, '../server.mjs')],
  env: { ...process.env, HSIN_COMPANION_PORT: '8799', HSIN_VRM_MODEL: path.join(HERE, '../../vrm_player/public/models/test1.vrm') }, stderr: 'ignore' }));   // a port nobody listens on: never the running app
const call = async (a) => (await client.callTool({ name: 'perform', arguments: a })).content?.[0]?.text ?? '';
const nod = [{ t: 0 }, { t: 0.8, head: { nod: 10 } }, { t: 1.4 }];
let t = await call({ say: 'a', subtitle: 'a', emotion: 'neutral', posture: 'w_sit', keyframes: nod });
check(t.startsWith('[internal note') && t.includes('Shown') && t.includes('Posture now: w_sit'), `sit: ${t.slice(0, 110)}`);
t = await call({ say: 'b', subtitle: 'b', emotion: 'neutral', keyframes: nod });
check(t.includes('Posture now: w_sit'), `no posture given keeps it: ${t.slice(34, 110)}`);
check(!t.includes('Language'), 'no Language line the second time');
t = await call({ say: 'c', emotion: 'neutral', posture: 'cross_legged', keyframes: nod });          // test1: long skirt
check(t.includes('Not shown yet') && t.includes('skirt_spread'), `1st rejection goes back: ${t.slice(34, 120)}`);
t = await call({ say: 'c', emotion: 'neutral', posture: 'cross_legged', keyframes: nod });
check(t.includes('Shown') && t.includes('simplified') && t.includes('Posture now: seiza'), `2nd rejection -> safe version: ${t.slice(34, 120)}`);
t = await call({ say: 'd', emotion: 'neutral', posture: 'stand', keyframes: [{ t: 0 }, { t: 1, right_arm: { hand: 'head out beside' } }, { t: 1.6 }] });
check(t.includes('Shown') && !t.includes('simplified') && t.includes('Posture now: stand'), `back to normal: ${t.slice(34, 110)}`);
await client.close();
console.log(bad ? `${bad} FAILED` : 'all ok'); process.exit(bad ? 1 : 0);
