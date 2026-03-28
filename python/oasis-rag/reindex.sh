#!/bin/bash
# O.A.S.I.S. Mac Re-indexing Script

# Ensure we're in the correct directory if called from elsewhere
cd "$(dirname "$0")"

# 1. Mac 스레드 충돌 방지 설정 (Segmentation Fault 방지)
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# 2. 가상환경 활성화 (없을 경우 대비)
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "[OASIS] Virtual environment (venv) not found. Please create it first."
    exit 1
fi

# 3. 인덱싱 실행
echo "[OASIS] Starting indexing..."
python indexer.py

echo ""
echo "[OASIS] Indexing complete! Please restart app.py to apply changes."
