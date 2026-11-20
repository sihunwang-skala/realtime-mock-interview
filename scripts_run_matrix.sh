#!/bin/bash
# Reproduces results/matrix_*.json (Case B). Run on an otherwise idle machine; Ollama must be serving.
cd "$(dirname "$0")"; PY=.venv/bin/python
L=${LLM:-ollama:qwen2.5:7b}
for st in two_track hybrid direct precomputed; do
  $PY -W ignore -m qa_lab.benchmark.run --case B --llm $L --strategy $st --interjection hybrid --out results/matrix_B_${st}_hybrid.json > results/log_B_${st}.txt 2>&1
done
for ij in rule llm; do
  $PY -W ignore -m qa_lab.benchmark.run --case B --llm $L --strategy two_track --interjection $ij --out results/matrix_B_two_track_${ij}.json > results/log_B_ij_${ij}.txt 2>&1
done
# Case A over a real LiveKit room (same scenarios)
$PY -W ignore -m qa_lab.benchmark.run_livekit --llm $L --strategy two_track --interjection hybrid --out results/case_A_two_track_hybrid.json > results/log_A.txt 2>&1
# LLM comparison (fast track model), two-track + hybrid interjection
for m in llama3.2:3b qwen3.5:4b llama3.1:8b; do
  n=$(echo $m | tr ':.' '__')
  $PY -W ignore -m qa_lab.benchmark.run --case B --llm ollama:$m --strategy two_track --interjection hybrid --out results/llm_B_$n.json > results/log_llm_$n.txt 2>&1
done
echo ALLDONE > results/matrix_done.txt
