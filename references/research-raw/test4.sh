#!/bin/bash
q="test"
out="$(python3 -c "import json,sys; print(json.dumps({'a': sys.argv[1]}))" "$q")"
echo "RESULT: $out"
