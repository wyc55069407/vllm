"""
Redirected: TopK tests have moved to tests/test_topk.py.
Run: python tests/test_topk.py
"""
import sys
import os

# Redirect to tests/test_topk.py
sys.path.insert(0, os.path.dirname(__file__))
exec(open(os.path.join(os.path.dirname(__file__), "tests", "test_topk.py")).read())
