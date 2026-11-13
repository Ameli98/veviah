"""Step 7: reward model v0 = small Transformer over joint sequences + instruction embedding.

Input : motion [T, J, 4] quaternions (15 fps after downsampling, cropped/padded to T_MAX) + text vector
Output: 3 scores (match, natural, appropriate); overall = their sum
Loss  : Bradley-Terry on preference pairs (+ small regression on absolute scores)
"""
import json
import re
import zlib
from pathlib import Path

import numpy as np

from .prefs import BROKEN_NATURAL
from .rotations import angular_velocity, to_6d
from .schema import J, JIDX, load_motion, load_root, read_jsonl
from .skeleton import HIPS_REST, KEY_POINTS, key_points

DS, T_MAX, FEAT = 2, 150, J * (6 + 3 + 3) + 6   # per joint: 6D rotation, angular velocity, its change; hips offset and its velocity
POS_FEAT = len(KEY_POINTS) * 6                      # optional: world positions of head, elbows, palms, knees, ankles + velocity
AXES = ["match", "natural", "appropriate"]


# ---------- text encoders ----------
class HashTextEncoder:
    """Bag-of-words hashed vectors. No semantics: a stand-in for tests."""
    dim = 64
    name = "hash"

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"[a-z']+", t.lower()) or ["_"]
            v = np.mean([np.random.default_rng(zlib.crc32(w.encode())).normal(size=self.dim) for w in toks], 0)
            out[i] = v / (np.linalg.norm(v) + 1e-8)
        return out


class SentenceTextEncoder:
    """Real encoder (pip install sentence-transformers). Untested in this repo's sandbox."""
    name = "minilm"

    def __init__(self, model="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)
        self.dim = self.m.get_sentence_embedding_dimension()

    def encode(self, texts):
        return np.asarray(self.m.encode(list(texts), normalize_embeddings=True), np.float32)


def get_encoder(name):
    return {"hash": HashTextEncoder, "minilm": SentenceTextEncoder}[name]()


# ---------- features ----------
def featurize(motion, root=None, pos=False):
    """root: hips offset [T, 3] (schema); None = standing still. pos: append KEY_POINTS positions
    (metres from the rest hips) and their velocity, so where the hands are does not have to be
    worked out from a chain of rotations."""
    m = np.asarray(motion[::DS], float)
    r = np.zeros((len(m), 3)) if root is None else np.asarray(root[::DS], float)
    rv = np.diff(r, axis=0, prepend=r[:1])
    v = np.zeros(m.shape[:2] + (3,))
    if len(m) > 1:
        v[1:] = angular_velocity(m, 1.0)       # radians per (downsampled) frame
    a = np.diff(v, axis=0, prepend=v[:1])      # acceleration exposes jitter
    x = np.concatenate([to_6d(m).reshape(len(m), -1), (v * 5).reshape(len(m), -1), (a * 20).reshape(len(m), -1),
                        r * 3, rv * 30], 1)
    if pos:
        k = key_points(np.asarray(motion, float), np.zeros((len(motion), 3)) if root is None else np.asarray(root, float), JIDX)[::DS]
        kv = np.diff(k, axis=0, prepend=k[:1])
        x = np.concatenate([x, ((k - HIPS_REST) * 2).reshape(len(m), -1), (kv * 10).reshape(len(m), -1)], 1)
    L = min(len(x), T_MAX)
    out = np.zeros((T_MAX, x.shape[1]), np.float32)
    out[:L] = x[:L]
    mask = np.zeros(T_MAX, bool)
    mask[:L] = True
    return out, mask


def _torch():
    import torch
    return torch


def build_model(text_dim, d=128, layers=3, heads=4, feat=FEAT):
    torch = _torch()
    nn = torch.nn

    class RewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp, self.txt = nn.Linear(feat, d), nn.Linear(text_dim, d)
            self.pos = nn.Parameter(torch.randn(1, T_MAX + 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.head = nn.Linear(d, len(AXES))

        def forward(self, x, mask, t):
            tok = torch.cat([self.txt(t)[:, None], self.inp(x)], 1) + self.pos
            pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~mask], 1)
            return self.head(self.enc(tok, src_key_padding_mask=pad)[:, 0])

    return RewardModel()


def _batch(ids, cache, tvec, iids, device):
    torch = _torch()
    x = torch.from_numpy(np.stack([cache[i][0] for i in ids])).float().to(device)
    m = torch.from_numpy(np.stack([cache[i][1] for i in ids])).to(device)
    t = torch.from_numpy(np.stack([tvec[i] for i in iids])).float().to(device)
    return x, m, t


def pair_accuracy(model, pairs, cache, tvec, device="cpu", bs=128):
    torch = _torch()
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for s in range(0, len(pairs), bs):
            b = pairs[s:s + bs]
            iids = [p["instruction_id"] for p in b]
            sw = model(*_batch([p["winner"] for p in b], cache, tvec, iids, device)).sum(-1)
            sl = model(*_batch([p["loser"] for p in b], cache, tvec, iids, device)).sum(-1)
            ok += (sw > sl).sum().item()
            n += len(b)
    return ok / max(1, n)


def free_pool(cands, scores, group_ids):
    """{group: [candidate ids]} usable in judge-free pairs: normal, passed the filter, not scored as broken."""
    pool = {}
    for c in cands.values():
        s = scores.get(c["id"])
        if (c["instruction_id"] in group_ids and c["source"] == "normal" and c.get("filter") == "pass"
                and (s is None or s["natural"] > BROKEN_NATURAL)):
            pool.setdefault(c["instruction_id"], []).append(c["id"])
    return pool


def free_mismatch_pairs(pool, k, rng, posture=None, hard=0.0):
    """k pairs per group that need no judge: a candidate of the group beats a candidate of another
    group, under this group's instruction. With probability `hard` the other group has the same
    posture ({group: posture}), so standing vs sitting alone does not decide the pair."""
    gids = sorted(pool)
    same = {}
    if posture:
        for g in gids:
            same.setdefault(posture[g], []).append(g)
    out = []
    for g in gids:
        for _ in range(k):
            cand = same[posture[g]] if posture and rng.random() < hard and len(same[posture[g]]) > 1 else gids
            other = g
            while other == g:
                other = cand[rng.integers(len(cand))]
            out.append({"instruction_id": g, "winner": pool[g][rng.integers(len(pool[g]))],
                        "loser": pool[other][rng.integers(len(pool[other]))], "kind": "free_mismatch"})
    return out


# Degradations used for judge-free 'broken' pairs: the ones the author marked broken (2026-10-21 labels);
# mirror and speed changes she ranked but never marked broken, so they are left out.
FREE_DEGRADE_KINDS = ["noise", "extreme", "freeze_arm", "freeze_legs", "truncate"]


def free_degrade_pairs(pool, k, rng, kinds=FREE_DEGRADE_KINDS):
    """k pairs per group: a candidate beats a degraded copy of itself (id '<cid>~<kind>')."""
    out = []
    for g in sorted(pool):
        for _ in range(k):
            c = pool[g][rng.integers(len(pool[g]))]
            out.append({"instruction_id": g, "winner": c, "loser": f"{c}~{kinds[rng.integers(len(kinds))]}",
                        "kind": "free_degrade"})
    return out


def train_reward_model(workdir, epochs=20, batch=64, lr=3e-4, aux_weight=0.1, encoder="hash",
                 device=None, seed=0, log=print, d=128, layers=3, free_mismatch=0, hard=0.0, free_degrade=0, pos=False):
    """Judge-free extra pairs per train group, drawn anew each epoch: free_mismatch (with a fraction
    `hard` against the same posture, free_mismatch_pairs) and free_degrade (free_degrade_pairs)."""
    torch = _torch()
    F = torch.nn.functional
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    w = Path(workdir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    instr = {i["id"]: i["text"] for i in read_jsonl(w / "instructions.jsonl")}
    cands = {c["id"]: c for c in read_jsonl(w / "candidates.jsonl")}
    scores = {s["cand_id"]: s for s in read_jsonl(w / "scores.jsonl")}
    pairs = read_jsonl(w / "pairs.jsonl")
    split = json.loads((w / "split.json").read_text())
    train_ids, val_ids = set(split["train"]), set(split["val"])
    tr = [p for p in pairs if p["instruction_id"] in train_ids]
    va = [p for p in pairs if p["instruction_id"] in val_ids]
    if not tr or not va:
        raise RuntimeError(f"need train and val pairs (got {len(tr)}/{len(va)})")

    enc = get_encoder(encoder)
    ids_needed = {p[k] for p in tr + va for k in ("winner", "loser")}
    free = free_mismatch or free_degrade
    pool = free_pool(cands, scores, train_ids) if free else {}
    posture = {i["id"]: i["spec"].get("posture", "stand") for i in read_jsonl(w / "instructions.jsonl")}
    ids_needed |= {c for cs in pool.values() for c in cs}
    if free_degrade:
        ids_needed |= {f"{c}~{k}" for cs in pool.values() for c in cs for k in FREE_DEGRADE_KINDS}
    def feat(i):
        from .generators import degrade, degrade_root
        cid, _, kind = i.partition("~")
        m = load_motion(w, cands[cid]["motion_path"])
        r = load_root(w, cands[cid]["motion_path"], len(m))
        if kind:
            m, r = degrade(m, kind, np.random.default_rng(zlib.crc32(i.encode()))), degrade_root(r, kind)
        return tuple(featurize(m, r, pos))
    cache = {i: feat(i) for i in ids_needed}
    iid_list = sorted({p["instruction_id"] for p in tr + va} | set(pool))
    tvec = dict(zip(iid_list, enc.encode([instr[i] for i in iid_list])))

    model = build_model(enc.dim, d=d, layers=layers, feat=FEAT + pos * POS_FEAT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def targets(cid):  # (score-3)/2, NaN where unavailable (e.g. filter rejects)
        s = scores.get(cid)
        return [(s[a] - 3) / 2 for a in AXES] if s else [np.nan] * len(AXES)

    best, best_state, history = -1, None, []
    for ep in range(epochs):
        model.train()
        ep_pairs = (tr + free_mismatch_pairs(pool, free_mismatch, rng, posture, hard)
                    + free_degrade_pairs(pool, free_degrade, rng))
        order = rng.permutation(len(ep_pairs))
        total = 0.0
        for s in range(0, len(order), batch):
            b = [ep_pairs[i] for i in order[s:s + batch]]
            iids = [p["instruction_id"] for p in b]
            ow = model(*_batch([p["winner"] for p in b], cache, tvec, iids, device))
            ol = model(*_batch([p["loser"] for p in b], cache, tvec, iids, device))
            loss = -F.logsigmoid(ow.sum(-1) - ol.sum(-1)).mean()
            # a free pair's loser was never scored as it is (other instruction, or degraded): no target
            tgt = torch.tensor([targets(p["winner"]) for p in b] +
                               [[np.nan] * len(AXES) if p["kind"].startswith("free") else targets(p["loser"]) for p in b],
                               dtype=torch.float32, device=device)
            out = torch.cat([ow, ol])
            ok = ~torch.isnan(tgt)
            if ok.any():
                loss = loss + aux_weight * F.mse_loss(out[ok], tgt[ok])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        acc = pair_accuracy(model, va, cache, tvec, device)
        history.append({"epoch": ep, "train_loss": total / len(ep_pairs), "val_pair_acc": acc})
        log(f"epoch {ep:2d} loss {total / len(ep_pairs):.4f} val_pair_acc {acc:.3f}")
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"state": best_state, "text_dim": enc.dim, "encoder": encoder, "d": d, "layers": layers, "pos": pos}, w / "reward_model.pt")
    metrics = {"best_val_pair_acc": best, "train_pairs": len(tr), "val_pairs": len(va), "history": history}
    (w / "reward_model_metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


class RewardModelScorer:
    """Inference wrapper: scores raw motions against instruction texts -> [n, 3]."""

    def __init__(self, path, device="cpu"):
        torch = _torch()
        ck = torch.load(path, map_location=device)
        self.enc = get_encoder(ck["encoder"])
        self.pos = ck.get("pos", False)
        self.model = build_model(ck["text_dim"], d=ck.get("d", 128), layers=ck.get("layers", 3),
                                 feat=FEAT + self.pos * POS_FEAT)
        self.model.load_state_dict(ck["state"])
        self.model.eval().to(device)
        self.device = device

    def score(self, motions, texts, roots=None):
        torch = _torch()
        feats = [featurize(m, r, self.pos) for m, r in zip(motions, roots or [None] * len(motions))]
        x = torch.from_numpy(np.stack([f[0] for f in feats])).to(self.device)
        m = torch.from_numpy(np.stack([f[1] for f in feats])).to(self.device)
        t = torch.from_numpy(self.enc.encode(texts)).to(self.device)
        with torch.no_grad():
            return self.model(x, m, t).cpu().numpy()
