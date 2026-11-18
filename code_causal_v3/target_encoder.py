"""Target encoder for the latent Action layer (groups B and C).

Per window and per bowl: 21 frames of image-centre motion (dx, dy relative to the
first visible point, plus visibility), from RECORDED mask centres, never from the
replay or from executed actions.

  group B : L_rec + L_cons          (no verb information at all)
  group C : L_rec + L_cons + SupCon (same verb sequence pulled together)

The encoder is frozen after training, the decoder discarded, and z* written per
episode to <out>/<group>/<episode>.npz: z[N,K,64] (L2 norm 8), valid[N,K].
K follows the Action clause order (bowls sorted by segmentation id).

Usage: target_encoder.py --group B|C --out DIR [--steps 4000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

W = Path.home() / "pi05-ledger-A"
WINDOW = 21
LATENT = 64
PIXEL_SCALE = 50.0
NORM = 8.0
FAILURES = "--failures" in sys.argv


def load_windows():
    """x[M,21,3], plus (episode, frame, entity) and the verb-sequence string of each window."""
    index = json.loads((W / "inputs/index.json").read_text())["episodes"]
    manifest = json.loads((W / "inputs/index.json").read_text()).get("split_manifest")
    xs, keys, verbs, reliable, splits = [], [], [], [], []
    split_of = {}
    for split in ("train", "val"):
        for line in open(W / "dataset_A3" / f"{split}.jsonl"):
            row = json.loads(line)
            split_of[row["sample_id"]] = (split, row["memory_target"]["action_text"], row["action_label_reliable"])
    del manifest
    shapes = {}
    for meta in index:
        name = meta["name"]
        z = np.load(W / "inputs" / f"{name}.npz")
        reg = meta["registry"]
        sids = [str(s) for s in z["sids"]]
        bowls = sorted([s for s in sids if reg[s]["category"] == "bowl"], key=lambda s: reg[s]["segmentation_id"])
        columns = [sids.index(s) for s in bowls]
        centres = z["centres"][:, columns].astype(np.float32)
        visible = z["visibility"][:, columns].astype(bool)
        n = len(centres)
        shapes[name] = (n, len(bowls))
        for t in range(n):
            split, action, ok = split_of[f"{name}_{t:06d}"]
            clauses = action.split(" | ")
            if len(clauses) != len(bowls):
                raise ValueError(f"Clause/bowl count mismatch at {name}@{t}")
            for k, clause in enumerate(clauses):
                frames = np.arange(t - WINDOW + 1, t + 1)
                inside = frames >= 0
                vis = np.zeros(WINDOW, bool)
                vis[inside] = visible[frames[inside], k]
                pos = np.zeros((WINDOW, 2), np.float32)
                pos[inside] = centres[frames[inside], k]
                if vis.any():
                    pos = pos - pos[np.argmax(vis)]
                pos = np.where(vis[:, None], pos, 0.0) / PIXEL_SCALE
                xs.append(np.concatenate((pos, vis[:, None].astype(np.float32)), axis=1))
                keys.append((name, t, k))
                verbs.append(clause.split(": ", 1)[1])
                reliable.append(bool(ok))
                splits.append(split)
    # Injected-failure episodes (training only): tracks and labels of what really happened in the replay.
    by_name = {e["name"]: e for e in index}
    for path in sorted((Path.home() / "pi05-abc/dataset/failures/episodes").glob("*.npz")) if FAILURES else []:
        z = np.load(path)
        name = path.stem.replace("_empty_hand", "").replace("_drop", "")
        reg = by_name[name]["registry"]
        sids = [str(s) for s in z["sids"]]
        bowls = sorted([s for s in sids if reg[s]["category"] == "bowl"], key=lambda s: reg[s]["segmentation_id"])
        columns = [sids.index(s) for s in bowls]
        centres = z["centres"][:, columns].astype(np.float32)
        visible = z["visibility"][:, columns].astype(bool)
        n = len(centres)
        shapes[path.stem] = (n, len(bowls))
        for t in range(n):
            for k, clause in enumerate(str(z["action_text"][t]).split(" | ")):
                frames = np.arange(t - WINDOW + 1, t + 1)
                inside = frames >= 0
                vis = np.zeros(WINDOW, bool)
                vis[inside] = visible[frames[inside], k]
                pos = np.zeros((WINDOW, 2), np.float32)
                pos[inside] = centres[frames[inside], k]
                if vis.any():
                    pos = pos - pos[np.argmax(vis)]
                pos = np.where(vis[:, None], pos, 0.0) / PIXEL_SCALE
                xs.append(np.concatenate((pos, vis[:, None].astype(np.float32)), axis=1))
                keys.append((path.stem, t, k))
                verbs.append(clause.split(": ", 1)[1])
                reliable.append(bool(z["action_reliable"][t]))
                splits.append("train")
    return np.stack(xs).astype(np.float32), keys, verbs, np.array(reliable), np.array(splits), shapes


class Encoder(nnx.Module):
    def __init__(self, rngs):
        self.c1 = nnx.Conv(3, 64, kernel_size=(5,), rngs=rngs)
        self.c2 = nnx.Conv(64, 128, kernel_size=(5,), strides=(2,), rngs=rngs)
        self.c3 = nnx.Conv(128, 128, kernel_size=(5,), strides=(2,), rngs=rngs)
        self.out = nnx.Linear(128 * 6, LATENT, rngs=rngs)
        self.norm = nnx.LayerNorm(LATENT, rngs=rngs)

    def __call__(self, x):
        h = nnx.gelu(self.c1(x))
        h = nnx.gelu(self.c2(h))
        h = nnx.gelu(self.c3(h))
        return self.norm(self.out(h.reshape(h.shape[0], -1)))


class Decoder(nnx.Module):
    def __init__(self, rngs):
        self.a = nnx.Linear(LATENT, 256, rngs=rngs)
        self.b = nnx.Linear(256, WINDOW * 2, rngs=rngs)

    def __call__(self, z):
        return self.b(nnx.gelu(self.a(z))).reshape(-1, WINDOW, 2)


def augment(rng, x):
    """Same trajectory, different observation noise: jitter, offset, short occlusion."""
    jitter_rng, offset_rng, start_rng, length_rng = jax.random.split(rng, 4)
    vis = x[..., 2:3]
    jitter = jax.random.normal(jitter_rng, x[..., :2].shape) * (1.0 / PIXEL_SCALE)
    offset = jax.random.normal(offset_rng, (x.shape[0], 1, 2)) * (1.0 / PIXEL_SCALE)
    start = jax.random.randint(start_rng, (x.shape[0], 1), 0, WINDOW)
    length = jax.random.randint(length_rng, (x.shape[0], 1), 0, 4)
    frames = jnp.arange(WINDOW)[None]
    keep = ~((frames >= start) & (frames < start + length))
    new_vis = vis * keep[..., None]
    return jnp.concatenate(((x[..., :2] + jitter + offset) * new_vis, new_vis), axis=-1)


def supcon(a, b, labels, mask, temperature=0.1):
    z = jnp.concatenate((a, b), axis=0)
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = jnp.concatenate((labels, labels))
    m = jnp.concatenate((mask, mask))
    sim = z @ z.T / temperature
    n = z.shape[0]
    sim = jnp.where(jnp.eye(n, dtype=bool), -1e9, sim)
    log_prob = sim - jax.nn.logsumexp(sim, axis=1, keepdims=True)
    positive = (y[:, None] == y[None]) & ~jnp.eye(n, dtype=bool) & m[:, None] & m[None]
    per_anchor = jnp.sum(jnp.where(positive, log_prob, 0.0), axis=1) / jnp.maximum(jnp.sum(positive, axis=1), 1)
    has = jnp.any(positive, axis=1)
    return -jnp.sum(jnp.where(has, per_anchor, 0.0)) / jnp.maximum(jnp.sum(has), 1)


def knn_accuracy(train_z, train_y, test_z, test_y, k=5, chunk=512):
    train_z = train_z / np.linalg.norm(train_z, axis=1, keepdims=True)
    test_z = test_z / np.linalg.norm(test_z, axis=1, keepdims=True)
    correct = 0
    for i in range(0, len(test_z), chunk):
        top = np.argpartition(-(test_z[i:i + chunk] @ train_z.T), k, axis=1)[:, :k]
        for row, y in zip(top, test_y[i:i + chunk]):
            values, counts = np.unique(train_y[row], return_counts=True)
            correct += int(values[np.argmax(counts)] == y)
    return correct / max(len(test_z), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("B", "C"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rec-weight", type=float, default=100.0)
    parser.add_argument("--sup-weight", type=float, default=1.0)
    parser.add_argument("--failures", action="store_true", help="Include injected-failure training episodes")
    parser.add_argument("--save-params", help="Pickle of the frozen encoder parameters")
    args = parser.parse_args()
    use_sup = args.group == "C"

    x, keys, verbs, reliable, splits, shapes = load_windows()
    classes = {v: i for i, v in enumerate(sorted(set(verbs)))}
    y = np.array([classes[v] for v in verbs], np.int32)
    visible = x[..., 2].sum(axis=1) > 0
    train = np.flatnonzero((splits == "train") & visible)
    by_class = defaultdict(list)
    for i in train:
        by_class[y[i]].append(i)
    by_class = {c: np.array(v) for c, v in by_class.items()}
    class_ids = np.array(sorted(by_class))
    print(json.dumps({"group": args.group, "windows": len(x), "train_windows": len(train),
                      "verb_sequences": len(classes), "seed": args.seed}), flush=True)

    rngs = nnx.Rngs(args.seed)
    encoder, decoder = Encoder(rngs), Decoder(rngs)
    graph, params = nnx.split((encoder, decoder))
    schedule = optax.warmup_cosine_decay_schedule(1e-5, 1e-3, 200, args.steps, 1e-5)
    optimizer = optax.adamw(schedule, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    def loss_fn(params, batch, labels, mask, rng):
        enc, dec = nnx.merge(graph, params)
        r1, r2 = jax.random.split(rng)
        v1, v2 = augment(r1, batch), augment(r2, batch)
        z, z1, z2 = enc(batch), enc(v1), enc(v2)
        vis = batch[..., 2:3]
        rec = jnp.sum(jnp.square(dec(z) - batch[..., :2]) * vis) / jnp.maximum(jnp.sum(vis) * 2, 1)
        cons = jnp.mean(jnp.square(z1 - z2))
        sup = supcon(z1, z2, labels, mask) if use_sup else jnp.asarray(0.0)
        # rec is in (px/50)^2: 1.5 px rms is about 9e-4, so scale it up to compete.
        total = args.rec_weight * rec + cons + (args.sup_weight * sup if use_sup else 0.0)
        return total, {"rec_px": jnp.sqrt(rec) * PIXEL_SCALE, "cons": cons, "sup": sup}

    @jax.jit
    def step(params, opt_state, batch, labels, mask, rng):
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, labels, mask, rng)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    numpy_rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    for i in range(args.steps):
        # Balanced by verb sequence, so rare sequences are not drowned out by "still".
        chosen = numpy_rng.choice(class_ids, args.batch)
        idx = np.array([numpy_rng.choice(by_class[c]) for c in chosen])
        key, sub = jax.random.split(key)
        params, opt_state, aux = step(params, opt_state, jnp.asarray(x[idx]), jnp.asarray(y[idx]),
                                      jnp.asarray(reliable[idx]), sub)
        if i % 500 == 0 or i == args.steps - 1:
            print(json.dumps({"step": i, **{k: round(float(v), 4) for k, v in aux.items()}}), flush=True)

    enc, dec = nnx.merge(graph, params)
    encode = jax.jit(lambda b: enc(b))
    decode = jax.jit(lambda b: dec(b))
    zs, recs = [], []
    for i in range(0, len(x), 8192):
        z = encode(jnp.asarray(x[i:i + 8192]))
        recs.append(np.asarray(decode(z)))
        zs.append(np.asarray(z))
    zs, recs = np.concatenate(zs), np.concatenate(recs)

    # Quality checks required before freezing.
    vis = x[..., 2:3]
    moving = np.array([v != "still" and v != "occluded" for v in verbs])
    val = (splits == "val") & visible
    def rmse(mask):
        err = np.sum(np.square(recs[mask] - x[mask][..., :2]) * vis[mask])
        return float(np.sqrt(err / max(np.sum(vis[mask]) * 2, 1)) * PIXEL_SCALE)
    train_moving = np.flatnonzero((splits == "train") & visible & moving & reliable)
    val_moving = np.flatnonzero(val & moving & reliable)
    sample = numpy_rng.choice(train_moving, min(40000, len(train_moving)), replace=False)
    report = {"group": args.group, "steps": args.steps, "verb_sequences": len(classes),
              "recon_rmse_px_val_all": rmse(val), "recon_rmse_px_val_moving": rmse(val & moving),
              "knn5_verb_sequence_acc_val_moving": knn_accuracy(zs[sample], y[sample], zs[val_moving], y[val_moving]),
              "majority_baseline_val_moving": float(np.max(np.bincount(y[val_moving])) / max(len(val_moving), 1))}
    print(json.dumps(report, indent=1), flush=True)

    out = args.out / args.group
    out.mkdir(parents=True, exist_ok=True)
    tables = {name: (np.zeros((n, k, LATENT), np.float32), np.zeros((n, k), bool)) for name, (n, k) in shapes.items()}
    for (name, t, k), z, ok in zip(keys, zs, visible):
        if ok:
            tables[name][0][t, k] = z / (np.linalg.norm(z) + 1e-8) * NORM
            tables[name][1][t, k] = True
    for name, (table, valid) in tables.items():
        np.savez_compressed(out / f"{name}.npz", z=table, valid=valid)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    if args.save_params:
        import pickle
        state = nnx.state(enc).to_pure_dict()
        with open(args.save_params, "wb") as stream:
            pickle.dump(jax.tree.map(np.asarray, state), stream)
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
