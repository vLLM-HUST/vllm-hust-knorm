#!/usr/bin/env bash
# KNorm server-verification experiment runner (how-to-run.md §2/§4/§7/§11).
# usage: run_experiment.sh <tag> <port> [knorm: off|on] [prefix: noprefix|prefix] [curl: nocurl|curl]
#
# This is the exact harness that produced the 2026-09-18 910B2 evidence
# (logs /tmp/vllm-knorm-<tag>.log), committed verbatim except: the
# readiness timeout now matches the 240-iteration loop, curl gets a
# --max-time ceiling, and paths/envs are overridable for non-server reuse.
set -u
TAG=$1; PORT=$2; KNORM=${3:-off}; PREFIX=${4:-noprefix}; DOCURL=${5:-nocurl}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/vllm-hust-dev/bin}
LOG=/tmp/vllm-knorm-${TAG}.log
MODEL=${MODEL:-/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8}
cd ~/vllm/vllm-hust-knorm   # avoid /root namespace-shadowing of vllm

PREFIX_FLAG=--no-enable-prefix-caching
[ "$PREFIX" = "prefix" ] && PREFIX_FLAG=--enable-prefix-caching
ENABLED=0; [ "$KNORM" = "on" ] && ENABLED=1

export HF_HUB_OFFLINE=1
export VLLM_VERSION=${VLLM_VERSION:-0.23.1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export VLLM_KNORM_ENABLED=$ENABLED
export VLLM_KNORM_COMPRESSION_RATIO=0.5
export VLLM_KNORM_WARMUP_TOKENS=32

nohup $PYTHON_BIN/vllm serve $MODEL \
  --served-model-name knorm-model \
  $PREFIX_FLAG \
  --max-model-len 8192 \
  --enforce-eager --gpu-memory-utilization 0.85 \
  --port $PORT \
  > $LOG 2>&1 &
SRV=$!
echo "server pid $SRV -> $LOG"

READY=0
for i in $(seq 1 240); do
  sleep 5
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER EXITED EARLY"; tail -30 $LOG; exit 1; fi
  if curl -s --max-time 10 -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200; then
    echo "READY after $((i*5))s"; READY=1; break
  fi
  if [ $i -eq 240 ]; then echo "TIMEOUT"; tail -30 $LOG; kill $SRV; exit 1; fi
done

if [ "$DOCURL" = "curl" ] && [ "$READY" = "1" ]; then
  echo "--- curl chat completion ---"
  curl -s --max-time 300 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "knorm-model", "messages": [{"role": "user", "content": "用一句话介绍武汉"}], "temperature": 0, "max_tokens": 64}' \
    -w "\nHTTP_CODE=%{http_code}\n"
fi

kill $SRV 2>/dev/null; sleep 5; kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "=== marker summary for $TAG ==="
grep -n "vllm-hust-knorm\|knorm-manager-registered\|knorm-wrapper-installed\|prefix caching is enabled" $LOG | head -20 || echo "(no knorm markers)"
echo "DONE $TAG"
