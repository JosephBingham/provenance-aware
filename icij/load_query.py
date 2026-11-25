import sys, time, pyoxigraph as ox
store_dir, nq, rq = sys.argv[1:4]
t=time.time(); st = ox.Store(store_dir)
if len(st) == 0:
    st.bulk_load(path=nq, format=ox.RdfFormat.N_QUADS)
print("quads", len(st), f"load {time.time()-t:.0f}s", flush=True)
t=time.time()
res = st.query(open(rq).read())
print([str(v) for v in res.variables])
for row in res: print([str(x) for x in row])
print(f"query {time.time()-t:.0f}s")
