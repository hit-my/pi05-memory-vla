"""first-frame candidates and appearance features (v5). Run-time inputs only: frame 0 RGB + SAM3 detections."""
from pathlib import Path

import cv2
import numpy as np

TV = Path("xxxx/rma/trackv")
YS, XS = np.mgrid[0:256, 0:256]
LO, HI = 20, 700


def load_det(ep, extra_dir=None, only_extra=False):
    out_m, out_s = [], []
    for d in ([] if only_extra else [TV / "initB/det"]) + ([Path(extra_dir)] if extra_dir else []):
        p = d / f"{ep}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        if z["masks"].size == 0:
            continue
        m = np.unpackbits(z["masks"], axis=-1)[..., :256].astype(bool).reshape(-1, 256, 256)
        out_m.append(m); out_s.append(z["scores"].astype(np.float32))
    if not out_m:
        return np.zeros((0, 256, 256), bool), np.zeros(0, np.float32)
    return np.concatenate(out_m), np.concatenate(out_s)


def candidates(ep, extra_dir=None, nms=0.7, only_extra=False):
    masks, scores = load_det(ep, extra_dir, only_extra)
    area = masks.sum(axis=(1, 2))
    ok = np.nonzero((area >= LO) & (area <= HI))[0]
    keep = []
    for i in ok[np.argsort(-scores[ok], kind="stable")]:
        if all((masks[i] & masks[j]).sum() / max(1, (masks[i] | masks[j]).sum()) <= nms for j in keep):
            keep.append(i)
    return masks[keep], scores[keep]


def feat(rgb, m):
    """appearance + shape features of one mask: [mean RGB(3), std RGB(3), hue hist(8) weighted by saturation, sat hist(3), val hist(3), area, w, h, fill]"""
    ys, xs = np.nonzero(m)
    px = rgb[ys, xs].astype(np.float32)
    hsv = cv2.cvtColor(px[None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0].astype(np.float32)
    hh = np.histogram(hsv[:, 0], bins=8, range=(0, 180), weights=hsv[:, 1] / 255.0)[0] / max(1, len(px))
    sh = np.histogram(hsv[:, 1], bins=3, range=(0, 256))[0] / max(1, len(px))
    vh = np.histogram(hsv[:, 2], bins=3, range=(0, 256))[0] / max(1, len(px))
    w, h = np.ptp(xs) + 1, np.ptp(ys) + 1
    return np.concatenate([px.mean(0) / 255, px.std(0) / 255, hh, sh, vh, [len(px) / 200.0, w / 20.0, h / 20.0, len(px) / float(w * h)]]).astype(np.float32)


def centroid(m):
    return np.array([XS[m].mean(), YS[m].mean()])


def bbox(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def biou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = ix * iy
    ua = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter
    return inter / ua if ua > 0 else 0.0


class Softmax:
    """multinomial logistic regression on standardised features (numpy; constants storable as JSON)"""

    def __init__(self, C=1.0, iters=3000, lr=0.5):
        self.C, self.iters, self.lr = C, iters, lr

    def fit(self, X, y, balanced=True):
        self.classes_ = np.unique(y)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        k = len(self.classes_)
        Y = (y[:, None] == self.classes_[None]).astype(np.float64)
        w = np.ones(n)
        if balanced:
            cnt = Y.sum(0)
            w = (n / (k * cnt))[np.argmax(Y, 1)]
        W, b = np.zeros((d, k)), np.zeros(k)
        for _ in range(self.iters):
            L = Z @ W + b
            L -= L.max(1, keepdims=True)
            P = np.exp(L); P /= P.sum(1, keepdims=True)
            G = (P - Y) * w[:, None] / w.sum()
            W -= self.lr * (Z.T @ G + W / (self.C * n))
            b -= self.lr * G.sum(0)
        self.W, self.b = W, b
        return self

    def predict_proba(self, X):
        L = ((np.asarray(X, np.float64) - self.mu) / self.sd) @ self.W + self.b
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)

    def to_json(self):
        return dict(classes=self.classes_.tolist(), mu=self.mu.tolist(), sd=self.sd.tolist(), W=self.W.tolist(), b=self.b.tolist())

    @classmethod
    def from_json(cls, d):
        s = cls()
        s.classes_, s.mu, s.sd, s.W, s.b = np.array(d["classes"]), np.array(d["mu"]), np.array(d["sd"]), np.array(d["W"]), np.array(d["b"])
        return s

def select_objects(P, K, pmin=0.5):
    """P [M, K+1] class probabilities of the M candidates (columns 0..K-1 = tracked objects, K = other).
    One candidate per object by assignment (maximum product of probabilities); an object whose candidate has P < pmin
    gets none (-1): it is not visible yet and is acquired later."""
    from scipy.optimize import linear_sum_assignment
    sel = [-1] * K
    if len(P) == 0:
        return sel
    cost = -np.log(P[:, :K].T + 1e-9)
    for a, b in zip(*linear_sum_assignment(cost)):
        if P[b, a] >= pmin:
            sel[a] = int(b)
    return sel

def contain(a, b):
    """share of bbox a (x0,y0,x1,y1 inclusive) that lies inside bbox b"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    return ix * iy / ((a[2] - a[0] + 1) * (a[3] - a[1] + 1))

def select_joint(P, cen, masks, K, pmin=0.5, dmax=25.0, lam=0.25, top=6, atyp=None, beta=0.0):
    """Joint choice of one candidate per object (or none): maximise  sum_k log P[i_k, k]  (an object without candidate costs
    log pmin)  -  lam * sum over chosen pairs of max(0, distance of their centroids - dmax)   [the tracked objects of these
    tasks stand next to each other on cabinet 1; dmax = learned largest usual distance],  no two choices on overlapping masks.
    Optional: + beta * log(min(1, area / atyp[k])) (a part of an object is a worse choice than the whole object).
    P [M, K+1], cen [M,2], masks bool [M,H,W]. Returns candidate index per object (-1 = none). Positions in the image are not used."""
    import itertools
    if len(P) == 0:
        return [-1] * K
    opts = []
    for k in range(K):
        idx = [int(i) for i in np.argsort(-P[:, k])[:top] if P[i, k] >= pmin]
        opts.append(idx + [-1])
    best, best_s = [-1] * K, -1e18
    for combo in itertools.product(*opts):
        ch = [i for i in combo if i >= 0]
        if len(set(ch)) < len(ch):
            continue
        s = sum(np.log(P[i, k] + 1e-9) if i >= 0 else np.log(pmin) - 1e-3 for k, i in enumerate(combo))
        if atyp is not None and beta > 0:  # prefer a detection of the whole object to one of a part of it (atyp = usual area of object k)
            s += beta * sum(np.log(min(1.0, float(masks[i].sum()) / atyp[k])) for k, i in enumerate(combo) if i >= 0)
        bad = False
        for a in range(K):
            for b in range(a + 1, K):
                i, j = combo[a], combo[b]
                if i < 0 or j < 0:
                    continue
                if (masks[i] & masks[j]).sum() > 0.3 * min(masks[i].sum(), masks[j].sum()):
                    bad = True
                s -= lam * max(0.0, float(np.hypot(*(cen[i] - cen[j]))) - dmax)
        if not bad and s > best_s:
            best, best_s = list(combo), s
    return best