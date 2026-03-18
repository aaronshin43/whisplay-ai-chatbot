"""Backward-compatibility stub — test suite moved to tests/run_all.py.

Usage (still works):
    python python/oasis-rag/validation/run_all.py
"""
import os
import runpy

_NEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "run_all.py")
runpy.run_path(os.path.normpath(_NEW), run_name="__main__")
