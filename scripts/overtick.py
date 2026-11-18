"""overtick.py META.json : per task, how far the model's checklist ran ahead of what the simulator scored.
ticked = highest 'done N' / checklist length ; real = CSR. An episode is 'over-ticked' if ticked - real >= 0.25.
Post-hoc analysis only (uses the scored stages, never fed to the model)."""
import json, re, sys
M = json.load(open(sys.argv[1], encoding='utf-8'))
T = {}
for m in M:
    if not m['memory']:
        continue
    g = re.search(r'init \[(.*)\]', m['memory'][0][1])
    total = len([x for x in g.group(1).split(';') if x.strip()]) if g else 0
    if total < 2 and g:
        total = len([x for x in g.group(1).split(',') if x.strip()])
    top = 0
    for c in m['memory']:
        for d in re.findall(r'Subgoal:\s*done\s+([\d,\s]+)', c[1]):
            top = max([top] + [int(x) for x in re.findall(r'\d+', d)])
    ticked = min(top / total, 1.0) if total else 0.0
    real = m['CSR'] / 100
    t = T.setdefault(m['task'], dict(n=0, ok=0, over=0, fail=0, tick=0.0, real=0.0, total=total))
    t['n'] += 1; t['ok'] += m['success']; t['tick'] += ticked; t['real'] += real
    if not m['success']:
        t['fail'] += 1; t['over'] += ticked - real >= 0.25
print('task items n succ | ticked% real% | over-ticked failures')
A = dict(n=0, ok=0, over=0, fail=0)
for k in sorted(T):
    t = T[k]
    print('T%-3d %d  %2d %2d | %3.0f %3.0f | %d/%d' % (k, t['total'], t['n'], t['ok'], 100 * t['tick'] / t['n'], 100 * t['real'] / t['n'], t['over'], t['fail']))
    for x in A: A[x] += t[x]
print('ALL n=%d succ=%d over-ticked failures %d/%d = %.0f%%' % (A['n'], A['ok'], A['over'], A['fail'], 100 * A['over'] / max(A['fail'], 1)))
