// The built-in chat window's bridge to main.js (chat_main.js).
const { contextBridge, ipcRenderer } = require('electron');
const call = (name) => (...a) => ipcRenderer.invoke(name, ...a);
contextBridge.exposeInMainWorld('hsinChat', { state: call('chatState'), send: call('chatSend'), clear: call('chatClear'), settings: call('openSettingsPage') });
