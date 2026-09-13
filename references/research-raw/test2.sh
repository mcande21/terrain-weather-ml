#!/bin/bash
set -uo pipefail
q="test query"
curl -s -X POST 'https://httpbin.org/post' \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "import json,sys; print(json.dumps({'query': sys.argv[1], 'type':'deep','num_results':8,'contents':{'highlights':{'max_characters':3000}}}))" "$q")" \
    -o /tmp/test2out.json --max-time 20
echo done
