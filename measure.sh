#!/bin/bash
# usage: ./measure.sh <threads> <tag> <text>
T=$1; TAG=$2; TEXT=$3
source .venv/bin/activate
OUT="outputs/${TAG}_t${T}"
python M5-TTS/infer_onnx.py --model weights --local-files-only --device cpu --threads $T --stream \
  --text "$TEXT" \
  --prompt-wav M5-TTS/assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output "${OUT}.wav" > "${OUT}.log" 2>&1 &
PID=$!
MAXCPU=0
while kill -0 $PID 2>/dev/null; do
  CPU=$(ps -o %cpu= -p $PID 2>/dev/null | tr -d ' ')
  [ -n "$CPU" ] && awk -v a="$MAXCPU" -v b="$CPU" 'BEGIN{exit !(b>a)}' && MAXCPU=$CPU
  sleep 0.2
done
wait $PID; RC=$?
python - "${OUT}.json" "$T" "$TAG" "$MAXCPU" $RC <<'PY'
import json,sys
p,t,tag,maxcpu,rc=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
try:
    d=json.load(open(p))
    print(f"{tag}\tt={t}\trtf={d['rtf']:.3f}\tfirst={d['first_pcm_s']:.3f}s\telapsed={d['elapsed_s']:.2f}s\taudio={d['audio_s']:.2f}s\tframes={d['frames']}\tmaxcpu={maxcpu}%\trc={rc}")
except Exception as e:
    print(f"{tag}\tt={t}\tERROR {e} rc={rc}")
PY
