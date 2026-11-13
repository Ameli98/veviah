// The avatar window's bridge (page = companion.js in app mode): click-through, dragging, zoom, right-click menu.
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('hsinApp', {
  setSolid: (on) => ipcRenderer.send('solid', !!on),
  dragStart: (x, y) => ipcRenderer.send('dragStart', x, y),
  dragMove: (x, y, groundY, box) => ipcRenderer.send('dragMove', x, y, groundY, box),
  onSnap: (cb) => ipcRenderer.on('snap', (_e, dy) => cb(dy)),
  dragEnd: (groundY, box) => ipcRenderer.send('dragEnd', groundY, box),
  onTarget: (cb) => ipcRenderer.on('target', (_e, t) => cb(t)),         // window snapping: what a drop here snaps to
  onAttach: (cb) => ipcRenderer.on('attach', (_e, t) => cb(t)),         // after the drop: snapped to t (or null)
  onDetach: (cb) => ipcRenderer.on('detach', () => cb()),               // the target window went away
  attached: (groundY, box) => ipcRenderer.send('attached', groundY, box),   // the new pose is shown: align to the edge
  fall: (groundY) => ipcRenderer.send('fall', groundY),
  follow: () => ipcRenderer.send('follow'),                             // once per drawn frame while snapped                 // drop onto the taskbar
  zoom: (f) => ipcRenderer.send('zoom', f),
  turn: (deg) => ipcRenderer.send('turn', deg),
  menu: () => ipcRenderer.send('menu'),
  onHint: (cb) => ipcRenderer.on('hint', (_e, text) => cb(text)),       // a short tip in the bubble (first runs)
  onJiggle: (cb) => ipcRenderer.on('jiggle', (_e, level) => cb(level)),
  log: (...a) => ipcRenderer.send('pagelog', ...a.map(String)),
});
