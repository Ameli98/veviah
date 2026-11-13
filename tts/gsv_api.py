"""Start GPT-SoVITS's own api_v2.py with cudnn.benchmark kept off.

  python tts/gsv_api.py <GPT-SoVITS root> [api_v2 args: -a 127.0.0.1 -p 9880 -c config.yaml]

GPT-SoVITS's Accel module turns torch.backends.cudnn.benchmark on the first time it is imported (during the first
synthesis); with it on every new input length starts a cuDNN algorithm search: 45-70 s per sentence on the 4070
(measured 2026-11-05; an earlier voice project saw ~20 s), < 1 s with it off. The setter is replaced so it stays off; the
GPT-SoVITS checkout is not modified."""
import os
import runpy
import sys

import torch

if os.name == "nt":      # Windows: never let EcoQoS move the TTS onto the slow path (see motion_loop/perform.py)
    import ctypes, ctypes.wintypes as _w
    class _S(ctypes.Structure):
        _fields_ = [("Version", _w.ULONG), ("ControlMask", _w.ULONG), ("StateMask", _w.ULONG)]
    _k = ctypes.windll.kernel32; _k.GetCurrentProcess.restype = _w.HANDLE
    _k.SetProcessInformation.argtypes = [_w.HANDLE, ctypes.c_int, ctypes.c_void_p, _w.DWORD]
    _st = _S(1, 1, 0); _k.SetProcessInformation(_k.GetCurrentProcess(), 4, ctypes.byref(_st), ctypes.sizeof(_st))

_cls = type(torch.backends.cudnn)
_cls.benchmark = property(lambda self: torch._C._get_cudnn_benchmark(),
                          lambda self, v: torch._C._set_cudnn_benchmark(False))
torch._C._set_cudnn_benchmark(False)

# Reference clips are read with soundfile instead of torchaudio.load (2026-11-12, step 4): torchaudio 2.9+ decodes
# through torchcodec, which needs FFmpeg's shared libraries; on Windows there are none and every /tts failed 'Could not
# load libtorchcodec'. GPT-SoVITS's inference only reads wav references (TTS.py: torchaudio.load twice), which
# soundfile handles; same (channels, samples) float32 tensor and sample rate as torchaudio.load returned.
import soundfile as _sf
import torchaudio as _ta


def _load(path, *a, **k):
    x, sr = _sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(x.T.copy()), sr


_ta.load = _load

root = os.path.abspath(sys.argv[1])
os.chdir(root)                      # GPT-SoVITS resolves its pretrained paths relative to the repo root
sys.path[:0] = [root, os.path.join(root, "GPT_SoVITS")]
sys.argv = [os.path.join(root, "api_v2.py")] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
