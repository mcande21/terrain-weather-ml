import json, glob, os

qfile = "/private/tmp/claude-501/-Users-cooperanderson/f9f3d1e6-e2be-47fb-b0c7-e68098372ccb/scratchpad/search.sh"
results_dir = "/private/tmp/claude-501/-Users-cooperanderson/f9f3d1e6-e2be-47fb-b0c7-e68098372ccb/scratchpad/results"

files = sorted(glob.glob(os.path.join(results_dir, "q*.json")), key=lambda p: int(os.path.basename(p)[1:-5]))

for f in files:
    idx = os.path.basename(f)[1:-5]
    try:
        with open(f) as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"=== Query {idx}: PARSE ERROR {e} ===")
        continue
    results = data.get("results", [])
    print(f"\n=== Query {idx} ({len(results)} results) ===")
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        print(f"- {title}\n  {url}")
        hl = r.get("highlights", [])
        for h in hl[:2]:
            snippet = h.replace("\n", " ")[:500]
            print(f"    > {snippet}")
