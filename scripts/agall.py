
import json,glob,sys,re,collections
R=collections.defaultdict(lambda:[0,0.0,0.0])
for root in sys.argv[1:]:
  for f in [x for x in glob.glob(root+"/*/t*_s*/aggregate.json") if "failed" not in x and "oom" not in x]:
    t=int(re.search(r"/t(\d+)_s\d+/",f).group(1)); a=json.load(open(f)); n=a["num_episodes"]
    R[t][0]+=n; R[t][1]+=a["TSR"]*n; R[t][2]+=a["CSR"]*n
for t in sorted(R): n,a,b=R[t]; print(t,n,round(a/n),round(b/n))
N=sum(v[0] for v in R.values()); print("ALL episodes",N,"TSR",round(sum(v[1] for v in R.values())/N,1),"CSR",round(sum(v[2] for v in R.values())/N,1))
print("task-mean TSR",round(sum(v[1]/v[0] for v in R.values())/len(R),1),"CSR",round(sum(v[2]/v[0] for v in R.values())/len(R),1),"tasks",len(R))

