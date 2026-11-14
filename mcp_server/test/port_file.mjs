// Ports taken by other programs (2026-11-14): freePort skips a held port, the app's companion writes the port it got
// to <data>/config/port.json, and an MCP server started without HSIN_COMPANION_PORT (as an AI app starts it) finds the
// companion through that file. Own data folder (HSIN_DATA) and ports from 8890: never the user's app.
// Run: node test/port_file.mjs
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const data = fs.mkdtempSync(path.join(os.tmpdir(), 'veviah-port-'));
process.env.HSIN_DATA = data;
let fail = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fail++; };
const { freePort, companionPort } = await import('../model.mjs');

// another program holds 8890
const holder = net.createServer().listen(8890, '127.0.0.1');
await new Promise((r) => holder.once('listening', r));
const p = await freePort(8890);
ok(p === 8891, `8890 held -> freePort gives ${p}`);

// the companion on that port, as the app starts it (HSIN_PORT_FILE=1)
const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')],
  { env: { ...process.env, HSIN_COMPANION_PORT: String(p), HSIN_PORT_FILE: '1' }, stdio: ['ignore', 'ignore', 'pipe'] });
await new Promise((r) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) r(); }));
const file = path.join(data, 'config', 'port.json');
let written = null;
for (let i = 0; i < 20 && !written; i++) { try { written = JSON.parse(fs.readFileSync(file, 'utf-8')); } catch { await new Promise((r) => setTimeout(r, 100)); } }
ok(written?.port === p, `port.json says ${written?.port} (${file})`);
const env0 = process.env.HSIN_COMPANION_PORT; delete process.env.HSIN_COMPANION_PORT;
ok(companionPort() === p, `companionPort() without env reads ${companionPort()}`);
if (env0) process.env.HSIN_COMPANION_PORT = env0;

// an MCP server started like an AI app starts it (no HSIN_COMPANION_PORT): its client name reaches the companion
const env = { ...process.env }; delete env.HSIN_COMPANION_PORT;
const client = new Client({ name: 'port-test-app', version: '1' });
await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(HERE, '../server.mjs')], env, stderr: 'ignore' }));
await client.listTools();
let seen = null;
for (let i = 0; i < 30 && !seen; i++) {
  const r = await fetch(`http://127.0.0.1:${p}/clients`).then((x) => x.json()).catch(() => ({}));
  if (JSON.stringify(r).includes('port-test-app')) seen = true; else await new Promise((res) => setTimeout(res, 200));
}
ok(seen, 'the MCP server found the companion on the new port (its client name arrived)');

// a test companion (no HSIN_PORT_FILE) leaves the file alone
fs.writeFileSync(file, JSON.stringify({ port: 1234 }));
const q = await freePort(8900);
const comp2 = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env: { ...process.env, HSIN_COMPANION_PORT: String(q) }, stdio: ['ignore', 'ignore', 'pipe'] });
await new Promise((r) => comp2.stderr.on('data', (d) => { if (String(d).includes('open http')) r(); }));
await new Promise((r) => setTimeout(r, 300));
ok(JSON.parse(fs.readFileSync(file, 'utf-8')).port === 1234, 'a companion without HSIN_PORT_FILE does not write the file');

await client.close(); comp.kill(); comp2.kill(); holder.close();
fs.rmSync(data, { recursive: true, force: true });
console.log(fail ? `${fail} failed` : 'all ok');
process.exit(fail ? 1 : 0);
