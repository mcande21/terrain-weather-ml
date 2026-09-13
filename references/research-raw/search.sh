#!/bin/bash
set -uo pipefail
OUTDIR="/private/tmp/claude-501/-Users-cooperanderson/f9f3d1e6-e2be-47fb-b0c7-e68098372ccb/scratchpad/results"
mkdir -p "$OUTDIR"

declare -a QUERIES=(
"Swiss IMIS network SLF high elevation automatic weather stations data access API"
"SLF IMIS avalanche warning weather station network GeoJSON download historical data"
"NREL wind resource complex terrain meteorological tower dataset dense array"
"Wind farm meteorological mast data complex terrain research dataset public download"
"Reynolds Creek Experimental Watershed weather station network data access dense"
"Senator Beck Basin Study Area weather station network data access"
"Niwot Ridge LTER meteorological station network data climate"
"NEON National Ecological Observatory Network meteorological stations complex terrain"
"MeteoSwiss automatic weather station network data access API download"
"ZAMG Austria weather station network open data mountain"
"Meteo France SAFRAR SURFEX mountain reanalysis Alps snow weather data access"
"CROCUS SAFRAN reanalysis French Alps snow meteorological data download"
"ski resort weather station network open data snow depth temperature wind"
"dense mountain meteorological station network complex terrain machine learning dataset"
"Swiss MeteoSwiss IDAWEB historical weather data mountain stations"
"Austria mountain weather station network eHYD hydrographic data open"
"Colorado avalanche information center weather station network data"
"SNOTEL network USDA snow telemetry weather stations data access density"
"complex terrain wind field observation dataset training machine learning WindNinja",
"alpine meteorological network dense stations 1km spacing terrain"
)

i=0
for q in "${QUERIES[@]}"; do
  i=$((i+1))
  fname="$OUTDIR/q${i}.json"
  echo "Running query $i: $q"
  BODY="$(python3 -c "import json,sys; print(json.dumps({'query': sys.argv[1], 'type':'deep','num_results':8,'contents':{'highlights':{'max_characters':3000}}}))" "$q")"
  curl -s -X POST 'https://api.exa.ai/search' \
    -H "x-api-key: ${EXA_API_KEY}" \
    -H 'Content-Type: application/json' \
    -d "$BODY" \
    -o "$fname" --max-time 90
  sleep 1
done
echo "DONE"
