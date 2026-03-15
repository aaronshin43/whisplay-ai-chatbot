#!/bin/bash
# index_knowledge.sh
#
# Rebuilds the O.A.S.I.S. RAG knowledge index from data/knowledge/*.md
#
# Usage:
#   ./index_knowledge.sh                    # index data/knowledge/
#   ./index_knowledge.sh path/to/knowledge  # custom knowledge directory

set -e

source ~/.bashrc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KNOWLEDGE_DIR="${1:-$SCRIPT_DIR/data/knowledge}"
INDEXER="$SCRIPT_DIR/python/oasis-rag/indexer.py"

echo "===== O.A.S.I.S. Knowledge Indexer ====="
echo "Knowledge dir : $KNOWLEDGE_DIR"
echo "Indexer       : $INDEXER"
echo ""

# Verify indexer exists
if [ ! -f "$INDEXER" ]; then
  echo "ERROR: indexer.py not found at $INDEXER"
  echo "Make sure you are in the project root directory."
  exit 1
fi

# Verify knowledge directory exists
if [ ! -d "$KNOWLEDGE_DIR" ]; then
  echo "ERROR: Knowledge directory not found: $KNOWLEDGE_DIR"
  echo "Create the directory and add Markdown (.md) files to it."
  exit 1
fi

echo "Running Python RAG indexer..."
python3 "$INDEXER" "$KNOWLEDGE_DIR"

echo ""
echo "===== Indexing complete ====="
echo "Index artifacts saved to: $SCRIPT_DIR/data/rag_index/"
echo ""
echo "If the RAG service is already running, restart it to reload the index:"
echo "  sudo systemctl restart oasis-rag   (if using systemd)"
echo "  kill <RAG_PID> && python3 python/oasis-rag/service.py &  (if running manually)"
