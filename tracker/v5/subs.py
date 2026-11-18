import collections
import numpy as np
import ev5
for t in range(1, 27):
    seqs = collections.Counter()
    for ep in ev5.task_eps(t)[:100:7]:
        m = np.load(f"/data/lgd/rma/lr_eps/meta/{ep}.npz")
        sub = m["subtask"]
        text = {int(s): str(x) for s, x in zip(m["subtask_order"], m["subtask_texts"])}
        starts = [0] + [i for i in range(1, len(sub)) if sub[i] != sub[i - 1]]
        seqs[" > ".join(f"{text[int(sub[i])]}@{i}" if False else text[int(sub[i])] for i in starts)] += 1
    print(t, len(sub), list(m.keys()) if t == 1 else "")
    for s, c in seqs.most_common(3):
        print("   ", c, s)
