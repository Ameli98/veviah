// The settings window's bridge to main.js.
const { contextBridge, ipcRenderer } = require('electron');
const call = (name) => (...a) => ipcRenderer.invoke(name, ...a);
contextBridge.exposeInMainWorld('hsin', {
  state: call('state'), chooseModel: call('chooseModel'), useModel: call('useModel'),
  copyInstructions: call('copyInstructions'), copyText: call('copyText'),
  clients: call('clients'), connectClient: call('connectClient'), disconnectClient: call('disconnectClient'), connectClientFile: call('connectClientFile'),
  setOnTop: call('setOnTop'), setSpeech: call('setSpeech'), setUiLang: call('setUiLang'), setJiggle: call('setJiggle'), setAutostart: call('setAutostart'),
  chatView: call('chatView'), setChat: call('setChat'), chatTest: call('chatTest'), chatModels: call('chatModels'), openChat: call('openChat'), openCredits: call('openCredits'), getOllama: call('getOllama'),
  voiceApi: call('voiceApi'), voiceAudio: call('voiceAudio'), choosePath: call('choosePath'),
});
