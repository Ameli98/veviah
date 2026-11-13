"""FROZEN (2026-11-08): the keyframe row layout (keyframes._LAYOUT) changed after the research track was aborted; the
checkpoints in ~/hsin_runs/policy_b no longer match it. Kept for the record; its tests are skipped."""
"""Policy option B: instruction text (+ duration) -> keyframe parameters; keyframes.py solves the motion.

The policy writes the same keyframes an LLM writes (parse_keyframes rows), so everything downstream
(solver, filter, renderer, reward model) is unchanged, and `to_keyframes_json` gives the LLM's JSON format.

Sequence per motion (autoregressive, small causal Transformer, text vector frozen):
  tokens   [text+duration]  [posture]  [kf 0] ... [kf K-1]
  predicts  posture          kf 0       kf 1 ...  stop
A keyframe = time (fraction of the duration) + per body part (GROUPS) a gate "this part changes here" +
for the parts that change: continuous values (angles, arm directions) as a Gaussian with a predicted std and
one categorical per word field (hand shapes, palms, leg words). A part that does not change keeps the
previous keyframe's values (the first keyframe is compared with the posture's rest), as an LLM leaves a
field out. Without the gates v0 resampled every part at every keyframe and the whole body drifted
(the author's blind test 2026-10-23: Haiku 15, B 0, tie 5). Leg parts are not trained (masked) when sitting,
since the solver ignores them. Sampling (temperature) gives different candidates for best-of-n;
temperature 0 = means, argmax and gates > 0.5.
"""
import json
from pathlib import Path

import numpy as np

from . import keyframes as kf
from .reward_model import _torch, get_encoder
from .schema import HAND_SHAPES as _ALL_SHAPES
HAND_SHAPES = _ALL_SHAPES[:6]          # the shapes its checkpoints know ('flat' came later, 2026-11-06)
LIFTS = kf.LIFTS[:4]       # the policy was trained before "heel" (2026-10-28); kept so its checkpoints still load
PLACES = kf.PLACES[:6]     # the checkpoints know these (cross was added 2026-10-29)

MAX_K = 16
MIN_K = 3
MIN_GAP = 0.1                      # seconds between keyframes when decoding
DUR_SCALE = 8.0
# continuous columns of a parse_keyframes row and their scale (value / scale ~ unit size)
CONT = [(i, 30.0) for i in range(6)] + [(i, 1.0) for i in range(6, 18)] + [(18, 20.0), (19, 20.0),
        (24, 40.0), (25, 40.0), (kf.TURN_AT, 45.0)]
CONT_COLS = np.array([c for c, _ in CONT])
CONT_SCALE = np.array([s for _, s in CONT])
N_CONT = len(CONT) + 1             # + time
# categorical columns: (column, options)
CATS = [(20, HAND_SHAPES), (21, HAND_SHAPES), (22, kf.PALMS), (23, kf.PALMS),
        (kf.LEG_AT, kf.WEIGHTS), (kf.LEG_AT + 1, kf.HEIGHTS), (kf.LEG_AT + 2, kf.STANCES),
        (kf.LEG_AT + 3, PLACES), (kf.LEG_AT + 4, LIFTS), (kf.LEG_AT + 5, PLACES), (kf.LEG_AT + 6, LIFTS),
        (kf.TURN_AT + 1, kf.LIFT_DIRS), (kf.TURN_AT + 2, kf.LIFT_DIRS)]
CAT_SIZES = [len(o) for _, o in CATS]
CAT_OFF = np.concatenate([[0], np.cumsum(CAT_SIZES)])
LEG_CAT = np.array([c >= kf.LEG_AT for c, _ in CATS])                  # masked when sitting
LEG_CONT = np.concatenate([[False], CONT_COLS == kf.TURN_AT])          # time (first) is never masked
IN_DIM = N_CONT + int(CAT_OFF[-1])
NP = len(kf.POSTURES)
# body parts that change together; columns of a parse_keyframes row
GROUPS = {"head": [0, 1, 2], "torso": [3, 4, 5], "right_arm": list(range(6, 12)), "left_arm": list(range(12, 18)),
          "shoulders": [18, 19], "hands": [20, 21], "palms": [22, 23], "wrists": [24, 25],
          "legs": list(range(kf.LEG_AT, kf.LEG_AT + 7)) + [kf.TURN_AT + 1, kf.TURN_AT + 2], "turn": [kf.TURN_AT]}
NG = len(GROUPS)
LEG_GROUP = np.array([g in ("legs", "turn") for g in GROUPS])              # masked when sitting
G_CONT = np.array([[i > 0 and CONT_COLS[i - 1] in cols for i in range(N_CONT)] for cols in GROUPS.values()])
G_CAT = np.array([[c in cols for c, _ in CATS] for cols in GROUPS.values()])


def gate_targets(posture, rows):
    """[K, NG] bool: does each part differ from the previous keyframe (the first one: from the rest row)."""
    prev = np.vstack([kf._rest_row(posture)[None], rows[:-1]])
    return np.stack([np.abs(rows[:, cols] - prev[:, cols]).max(1) > 1e-6 for cols in GROUPS.values()], 1)


def encode_rows(times, rows, duration_s):
    """[K] times, [K, 36] rows -> cont [K, N_CONT] (time first), cats [K, len(CATS)] int."""
    cont = np.concatenate([np.asarray(times)[:, None] / duration_s, rows[:, CONT_COLS] / CONT_SCALE], 1)
    cats = np.stack([rows[:, c] for c, _ in CATS], 1).astype(np.int64)
    return cont.astype(np.float32), cats


def token_features(cont, cats):
    onehot = np.zeros((len(cats), int(CAT_OFF[-1])), np.float32)
    for j in range(len(CATS)):
        onehot[np.arange(len(cats)), CAT_OFF[j] + cats[:, j]] = 1
    return np.concatenate([cont, onehot], 1)


def decode_rows(posture, cont, cats, duration_s):
    """Inverse of encode_rows. Arm directions are renormalised; times made increasing and inside the duration."""
    K = len(cont)
    rows = np.tile(kf._rest_row(posture), (K, 1))
    rows[:, CONT_COLS] = cont[:, 1:] * CONT_SCALE
    for at in (6, 9, 12, 15):                        # arm directions
        rows[:, at:at + 3] = kf._unit(rows[:, at:at + 3])
    for j, (c, _) in enumerate(CATS):
        rows[:, c] = cats[:, j]
    t = np.clip(cont[:, 0], 0, 1) * duration_s
    t[0] = 0.0
    for k in range(1, K):
        t[k] = max(t[k], t[k - 1] + MIN_GAP)
    if t[-1] > duration_s:                           # the gaps pushed it past the end: squeeze back
        t *= duration_s / t[-1]
    return t, rows


def to_keyframes_json(posture, times, rows):
    """Rows -> the JSON format the LLM writes (every field explicit), so parse_keyframes reads it back."""
    out = []
    for t, r in zip(times, rows):
        k = {"t": round(float(t), 3),
             "head": {n: round(float(r[i]), 1) for i, n in enumerate(kf.HEAD_LIM)},
             "torso": {n: round(float(r[3 + i]), 1) for i, n in enumerate(kf.TORSO_LIM)},
             "shoulders": {"right": round(float(r[18]), 1), "left": round(float(r[19]), 1)},
             "hands": {"left": HAND_SHAPES[int(r[20])], "right": HAND_SHAPES[int(r[21])]}}
        for i, (side, at) in enumerate((("right_arm", 6), ("left_arm", 12))):
            k[side] = {"upper": [round(float(x), 3) for x in r[at:at + 3]],
                       "fore": [round(float(x), 3) for x in r[at + 3:at + 6]],
                       "palm": kf.PALMS[int(r[22 + i])], "wrist": round(float(r[24 + i]), 1)}
        if posture == "stand":
            L = kf.LEG_AT
            k["legs"] = {"weight": kf.WEIGHTS[int(r[L])], "height": kf.HEIGHTS[int(r[L + 1])],
                         "stance": kf.STANCES[int(r[L + 2])], "turn": round(float(r[kf.TURN_AT]), 1)}
            for i, side in enumerate(("left", "right")):
                k["legs"][f"{side}_foot"] = {"place": PLACES[int(r[L + 3 + 2 * i])], "lift": LIFTS[int(r[L + 4 + 2 * i])],
                                             "dir": kf.LIFT_DIRS[int(r[kf.TURN_AT + 1 + i])]}
            if k["legs"]["height"] == "tiptoe":      # the word is gone (2026-10-28): heels up on the feet that are down
                k["legs"]["height"] = "straight"
                for side in ("left", "right"):
                    if k["legs"][f"{side}_foot"]["lift"] == "down":
                        k["legs"][f"{side}_foot"]["lift"] = "heel"
        out.append(k)
    return {"posture": posture, "keyframes": out}


def build_policy(text_dim, d=128, layers=3, heads=4):
    torch = _torch()
    nn = torch.nn

    class KeyframePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.cond = nn.Linear(text_dim + 1, d)
            self.post_in = nn.Embedding(NP, d)
            self.kf_in = nn.Linear(IN_DIM, d)
            self.pos = nn.Parameter(torch.randn(1, MAX_K + 2, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.post_head = nn.Linear(d, NP)
            self.mu, self.logstd = nn.Linear(d, N_CONT), nn.Linear(d, N_CONT)
            self.cat_head = nn.Linear(d, int(CAT_OFF[-1]))
            self.stop_head = nn.Linear(d, 1)
            self.gate_head = nn.Linear(d, NG)

        def forward(self, text, dur, posture, x):
            """text [B, D], dur [B], posture [B] long, x [B, k, IN_DIM] (k >= 0 keyframes so far).
            Returns the hidden states [B, k + 2, d]."""
            tok = torch.cat([self.cond(torch.cat([text, dur[:, None] / DUR_SCALE], 1))[:, None],
                             self.post_in(posture)[:, None], self.kf_in(x)], 1)
            n = tok.shape[1]
            causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tok.device), 1)
            return self.tf(tok + self.pos[:, :n], mask=causal)

    return KeyframePolicy()


def load_sft(path):
    """sft_candidates.jsonl -> list of dicts with text, duration, posture, cont, cats, split."""
    data = []
    for line in Path(path).read_text().splitlines():
        c = json.loads(line)
        d = json.loads(Path(c["keyframes_path"]).read_text())
        posture, times, rows = kf.parse_keyframes(d["keyframes"], d["duration_s"])
        if len(times) > MAX_K:
            continue
        cont, cats = encode_rows(times, rows, d["duration_s"])
        data.append(dict(id=c["id"], instruction_id=c["instruction_id"], text=c["text"], duration=d["duration_s"],
                         posture=kf.POSTURES.index(posture), cont=cont, cats=cats, gates=gate_targets(posture, rows),
                         split=c["split"]))
    return data


def _batch(items, tvec, device):
    torch = _torch()
    B, K = len(items), max(len(it["cont"]) for it in items)
    x = np.zeros((B, K, IN_DIM), np.float32)
    cont = np.zeros((B, K, N_CONT), np.float32)
    cats = np.zeros((B, K, len(CATS)), np.int64)
    gates = np.zeros((B, K, NG), np.float32)
    valid = np.zeros((B, K + 1), bool)               # targets: kf 0..K-1 and the stop after the last one
    for b, it in enumerate(items):
        k = len(it["cont"])
        x[b, :k] = token_features(it["cont"], it["cats"])
        cont[b, :k], cats[b, :k], gates[b, :k] = it["cont"], it["cats"], it["gates"]
        valid[b, :k + 1] = True
    stop = np.zeros((B, K + 1), np.float32)
    stop[np.arange(B), [len(it["cont"]) for it in items]] = 1
    f = lambda a: torch.from_numpy(a).to(device)
    return (f(np.stack([tvec[it["id"]] for it in items])), f(np.array([it["duration"] for it in items], np.float32)),
            f(np.array([it["posture"] for it in items])), f(x), f(cont), f(cats), f(gates), f(valid), f(stop))


def policy_loss(model, batch):
    torch = _torch()
    F = torch.nn.functional
    text, dur, posture, x, cont, cats, gates, valid, stop = batch
    h = model(text, dur, posture, x)                  # [B, K+2, d]
    lp = F.cross_entropy(model.post_head(h[:, 0]), posture)
    hk = h[:, 1:]                                     # predicts kf 0..K-1, then stop
    K = cont.shape[1]
    kmask = valid[:, :K].float()
    sitting = (posture != 0).float()[:, None, None]
    T = lambda a: torch.as_tensor(a, device=x.device).float()
    gmask = kmask[..., None] * (1 - sitting * T(LEG_GROUP))                     # [B, K, NG]
    lg = (F.binary_cross_entropy_with_logits(model.gate_head(hk[:, :K]), gates, reduction="none") * gmask).sum() / gmask.sum()
    changed = gates * gmask                                                     # only parts that change carry values
    mu, logstd = model.mu(hk[:, :K]), model.logstd(hk[:, :K]).clamp(-5, 2)
    nll = 0.5 * ((cont - mu) / logstd.exp()) ** 2 + logstd
    cmask = (changed @ T(G_CONT)).clamp(max=1)
    cmask[..., 0] = kmask                                                       # time: always
    lc = (nll * cmask).sum() / cmask.sum()
    catmask = (changed @ T(G_CAT)).clamp(max=1)
    logits = model.cat_head(hk[:, :K])
    lcat, n = 0.0, 0
    for j in range(len(CATS)):
        ce = F.cross_entropy(logits[..., CAT_OFF[j]:CAT_OFF[j + 1]].reshape(-1, CAT_SIZES[j]), cats[..., j].reshape(-1),
                             reduction="none").view_as(kmask)
        m = catmask[..., j]
        lcat = lcat + (ce * m).sum() / m.sum().clamp(min=1)
        n += 1
    ls = (F.binary_cross_entropy_with_logits(model.stop_head(hk).squeeze(-1), stop, reduction="none") * valid).sum() / valid.sum()
    return lp + lg + lc + lcat / n + ls, dict(posture=lp.item(), gate=lg.item(), cont=lc.item(), cat=(lcat / n).item(), stop=ls.item())


def train_policy(sft_path, out, epochs=200, batch=64, lr=3e-4, encoder="minilm", d=128, layers=3, seed=0,
                 device=None, log=print):
    """SFT on sft_candidates.jsonl (split column). Keeps the checkpoint with the lowest val loss."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = load_sft(sft_path)
    tr, va = [x for x in data if x["split"] == "train"], [x for x in data if x["split"] == "val"]
    enc = get_encoder(encoder)
    tvec = dict(zip([x["id"] for x in data], enc.encode([x["text"] for x in data])))
    model = build_policy(enc.dim, d=d, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, best_state, history = float("inf"), None, []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        for s in range(0, len(order), batch):
            loss, _ = policy_loss(model, _batch([tr[i] for i in order[s:s + batch]], tvec, device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = [policy_loss(model, _batch(va[s:s + batch], tvec, device)) for s in range(0, len(va), batch)]
        v = float(np.mean([l.item() for l, _ in vl]))
        history.append(dict(epoch=ep, val=v, **vl[0][1]))
        if v < best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"epoch {ep} val {v:.3f} best {best:.3f}")
    torch.save(dict(state=best_state, encoder=encoder, text_dim=enc.dim, d=d, layers=layers, history=history,
                    n_train=len(tr), n_val=len(va)), out)
    return dict(best_val=best, n_train=len(tr), n_val=len(va))


class KeyframePolicySampler:
    """Inference: sample(texts, durations, n, temperature) -> per text a list of n keyframe JSON objects.
    postures: optional posture name per text; given, it is used instead of the predicted one (in the app the
    character's current posture is known state, and a motion keeps one posture)."""

    def __init__(self, path, device="cpu"):
        torch = _torch()
        ck = torch.load(path, map_location=device)
        self.enc = get_encoder(ck["encoder"])
        self.model = build_policy(ck["text_dim"], d=ck["d"], layers=ck["layers"])
        self.model.load_state_dict(ck["state"])
        self.model.eval().to(device)
        self.device = device

    def sample(self, texts, durations, n=1, temperature=1.0, seed=0, postures=None):
        torch = _torch()
        gen = torch.Generator(device=self.device).manual_seed(seed)
        tv = torch.from_numpy(self.enc.encode(list(texts))).to(self.device)
        dur = torch.tensor(list(durations), dtype=torch.float32, device=self.device)
        text, dur = tv.repeat_interleave(n, 0), dur.repeat_interleave(n, 0)
        B, m = len(text), self.model
        with torch.no_grad():
            h = m(text, dur, torch.zeros(B, dtype=torch.long, device=self.device), torch.zeros(B, 0, IN_DIM, device=self.device))
            posture = self._pick(m.post_head(h[:, 0]), temperature, gen)
            if postures is not None:
                posture = torch.tensor([kf.POSTURES.index(p) for p in postures], device=self.device).repeat_interleave(n, 0)
            x = torch.zeros(B, 0, IN_DIM, device=self.device)
            rest = [encode_rows([0.0], kf._rest_row(p)[None], 1.0) for p in kf.POSTURES]
            prev_cont = torch.from_numpy(np.stack([rest[int(p)][0][0] for p in posture])).to(self.device)
            prev_cats = torch.from_numpy(np.stack([rest[int(p)][1][0] for p in posture])).to(self.device)
            g_cont, g_cat = (torch.as_tensor(a, device=self.device).float() for a in (G_CONT, G_CAT))
            conts, catss, done, length = [], [], torch.zeros(B, dtype=torch.bool, device=self.device), torch.full((B,), MAX_K, device=self.device)
            for k in range(MAX_K):
                hk = m(text, dur, posture, x)[:, -1]
                if k >= MIN_K:
                    p_stop = torch.sigmoid(m.stop_head(hk).squeeze(-1))
                    stop = (p_stop > 0.5) if temperature == 0 else (torch.rand(B, generator=gen, device=self.device) < p_stop)
                    length = torch.where(stop & ~done, torch.full_like(length, k), length)
                    done |= stop
                    if done.all():
                        break
                mu, std = m.mu(hk), m.logstd(hk).clamp(-5, 2).exp()
                cont = mu + temperature * std * torch.randn(mu.shape, generator=gen, device=self.device)
                logits = m.cat_head(hk)
                cats = torch.stack([self._pick(logits[:, CAT_OFF[j]:CAT_OFF[j + 1]], temperature, gen)
                                    for j in range(len(CATS))], 1)
                p_gate = torch.sigmoid(m.gate_head(hk))
                gate = ((p_gate > 0.5) if temperature == 0 else (torch.rand(p_gate.shape, generator=gen, device=self.device) < p_gate)).float()
                keep_c = (gate @ g_cont).clamp(max=1) < 0.5
                keep_c[:, 0] = False                                         # time is always new
                cont = torch.where(keep_c, prev_cont, cont)
                cats = torch.where((gate @ g_cat).clamp(max=1) < 0.5, prev_cats, cats)
                prev_cont, prev_cats = cont, cats
                conts.append(cont), catss.append(cats)
                feat = torch.from_numpy(token_features(cont.cpu().numpy(), cats.cpu().numpy())).to(self.device)
                x = torch.cat([x, feat[:, None]], 1)
        cont, cats = torch.stack(conts, 1).cpu().numpy(), torch.stack(catss, 1).cpu().numpy()
        out = []
        for b in range(B):
            p = kf.POSTURES[int(posture[b])]
            L = int(length[b])
            t, rows = decode_rows(p, cont[b, :L], cats[b, :L], float(dur[b]))
            out.append(to_keyframes_json(p, t, rows))
        return [out[i * n:(i + 1) * n] for i in range(len(texts))]

    @staticmethod
    def _pick(logits, temperature, gen):
        torch = _torch()
        if temperature == 0:
            return logits.argmax(-1)
        return torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).squeeze(-1)
