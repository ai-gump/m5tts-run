#!/bin/bash
# usage: ./measure_ac.sh <threads> <mode> <tag> <text>
T=$1; M=$2; TAG=$3; TEXT=$4
BIN=./audio.cpp/build/bin/audiocpp_cli
OUT="outputs/ac_${TAG}"
$BIN --family m5_tts --task clon --backend metal --model m5-voiceclone-f16.gguf \
  --voice-ref M5-TTS/assets/aslp_default_voice.wav \
  --reference-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --language zh --text "$TEXT" --threads $T --mode $M --out "${OUT}.wav" --metrics > "${OUT}.log" 2>&1 &
PID=$!
MAXCPU=0
while kill -0 $PID 2>/dev/null; do
  CPU=$(ps -o %cpu= -p $PID 2>/dev/null | tr -d ' ')
  [ -n "$CPU" ] && awk -v a="$MAXCPU" -v b="$CPU" 'BEGIN{exit !(b>a)}' && MAXCPU=$CPU
  sleep 0.2
done
wait $PID; RC=$?
RTF=$(grep "metrics.rtf=" "${OUT}.log" | cut -d= -f2)
WALL=$(grep "metrics.wall_ms=" "${OUT}.log" | cut -d= -f2)
AUD=$(grep "metrics.audio_duration_ms=" "${OUT}.log" | cut -d= -f2)
RSS=$(grep "peak_rss_mb=" "${OUT}.log" | cut -d= -f2)
echo "${TAG}\tthreads=${T}\tmode=${M}\trtf=${RTF}\twall=${WALL}ms\taudio=${AUD}ms\trss=${RSS}MB\tmaxcpu=${MAXCPU}%\trc=${RC}"
