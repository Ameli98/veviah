// Other programs' windows on Windows (2026-11-11, window snapping): top-level windows that are visible, not minimized,
// not cloaked (hidden UWP / other virtual desktops), not tool windows, with a title, in Z order (topmost first), as
// rectangles in DIP (Electron's screen coordinates). Uses user32 / dwmapi through koffi (npm, prebuilt for win32_x64).
const koffi = require('koffi');
const { screen } = require('electron');

const user32 = koffi.load('user32.dll');
const dwmapi = koffi.load('dwmapi.dll');
const RECT = koffi.struct('HSIN_RECT', { left: 'long', top: 'long', right: 'long', bottom: 'long' });
const EnumProc = koffi.proto('bool __stdcall HSIN_EnumProc(void *hwnd, intptr_t lparam)');
const EnumWindows = user32.func('bool __stdcall EnumWindows(HSIN_EnumProc *cb, intptr_t lparam)');
const IsWindowVisible = user32.func('bool __stdcall IsWindowVisible(void *hwnd)');
const IsIconic = user32.func('bool __stdcall IsIconic(void *hwnd)');
const IsZoomed = user32.func('bool __stdcall IsZoomed(void *hwnd)');
const GetWindowTextLengthW = user32.func('int __stdcall GetWindowTextLengthW(void *hwnd)');
const GetWindowLongPtrW = user32.func('intptr_t __stdcall GetWindowLongPtrW(void *hwnd, int index)');
const GetClassNameW = user32.func('int __stdcall GetClassNameW(void *hwnd, _Out_ uint16_t *buf, int max)');
const GetWindowRect = user32.func('bool __stdcall GetWindowRect(void *hwnd, _Out_ HSIN_RECT *rect)');
const DwmGetWindowAttribute = dwmapi.func('long __stdcall DwmGetWindowAttribute(void *hwnd, uint32_t attr, _Out_ void *value, uint32_t size)');
const GWL_EXSTYLE = -20, WS_EX_TOOLWINDOW = 0x80, WS_EX_LAYERED = 0x80000, DWMWA_EXTENDED_FRAME_BOUNDS = 9, DWMWA_CLOAKED = 14;
const SKIP_CLASS = new Set(['Progman', 'WorkerW', 'Shell_TrayWnd', 'Shell_SecondaryTrayWnd']);

function className(h) {
  const buf = new Uint16Array(256);
  const n = GetClassNameW(h, buf, 256);
  return String.fromCharCode(...buf.slice(0, n));
}
function rectOf(h) {
  const r = {};
  const b = Buffer.alloc(16);
  if (DwmGetWindowAttribute(h, DWMWA_EXTENDED_FRAME_BOUNDS, b, 16) === 0)   // the visible frame (no shadow margin)
    Object.assign(r, { left: b.readInt32LE(0), top: b.readInt32LE(4), right: b.readInt32LE(8), bottom: b.readInt32LE(12) });
  else if (!GetWindowRect(h, r)) return null;
  return r;
}

// own: a Set of window handles (BigInt) to leave out (HSIN's own windows)
function listWindows(own = new Set()) {
  const out = [];
  const cb = koffi.register((h) => {
    try {
      const id = koffi.address(h);
      if (own.has(id) || !IsWindowVisible(h) || IsIconic(h)) return true;
      const ex = Number(GetWindowLongPtrW(h, GWL_EXSTYLE));
      if (ex & WS_EX_TOOLWINDOW) return true;
      if (GetWindowTextLengthW(h) === 0 || SKIP_CLASS.has(className(h))) return true;
      const cl = Buffer.alloc(4);
      if (DwmGetWindowAttribute(h, DWMWA_CLOAKED, cl, 4) === 0 && cl.readUInt32LE(0) !== 0) return true;
      const r = rectOf(h);
      if (!r || r.right - r.left < 80 || r.bottom - r.top < 60) return true;
      const d = screen.screenToDipRect(null, { x: r.left, y: r.top, width: r.right - r.left, height: r.bottom - r.top });
      // full screen: covers a whole display (games, F11 browsers are not "maximized")
      const disp = screen.getDisplayMatching(d).bounds;
      const full = d.x <= disp.x && d.y <= disp.y && d.x + d.width >= disp.x + disp.width && d.y + d.height >= disp.y + disp.height;
      // an overlay: a layered (alpha) window spanning the whole display / work area, e.g. a screenshot tool's dimming
      // layer; it never covers an edge she sits on (2026-11-11: she fell off twice while the author took screenshots)
      const wa = screen.getDisplayMatching(d).workArea;
      const span = d.x <= wa.x + 2 && d.y <= wa.y + 2 && d.x + d.width >= wa.x + wa.width - 2 && d.y + d.height >= wa.y + wa.height - 2;
      const overlay = !!(ex & WS_EX_LAYERED) && span;
      out.push({ id: String(id), x: d.x, y: d.y, width: d.width, height: d.height, maximized: !!IsZoomed(h) || full, full, overlay, cls: className(h) });
    } catch { /* a window that went away meanwhile */ }
    return true;
  }, koffi.pointer(EnumProc));
  try { EnumWindows(cb, 0); } finally { koffi.unregister(cb); }
  return out;
}

// one window by its id (cheap: for following a window that is being moved); null = gone / hidden / minimized
const IsWindow = user32.func('bool __stdcall IsWindow(void *hwnd)');
function windowRect(id) {
  const h = BigInt(id);
  if (!IsWindow(h) || !IsWindowVisible(h) || IsIconic(h)) return null;
  const r = rectOf(h);
  if (!r) return null;
  const d = screen.screenToDipRect(null, { x: r.left, y: r.top, width: r.right - r.left, height: r.bottom - r.top });
  return { id: String(id), x: d.x, y: d.y, width: d.width, height: d.height, maximized: !!IsZoomed(h) };
}

module.exports = { listWindows, windowRect };
