"""Paths. Override with environment variables ICIJ_DIR (extracted CSV export), WORK_DIR (intermediate files,
several GB) and RESULTS_DIR."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICIJ = os.environ.get("ICIJ_DIR", os.path.join(ROOT, "data", "icij"))
WORK = os.environ.get("WORK_DIR", os.path.join(ROOT, "work"))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
