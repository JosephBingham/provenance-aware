"""Paths for the OpenSanctions experiment. Override with OS_EXTRACT_DIR (unzipped output of get_opensanctions.py),
OS_WORK_DIR (intermediate files of the plan's first cohort, built by build.py), OS_WORK_CELLS_DIR (intermediate files of
the count-invariant cohort reported in the paper, built by build2.py) and RESULTS_DIR. OS_COHORT selects the cohort that
load(), e2eval.py and the other analysis scripts use: "cells" (default, the paper's confirmatory cohort) or "first"."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT = os.environ.get("OS_EXTRACT_DIR", os.path.join(ROOT, "data", "opensanctions"))
WORK_FIRST = os.environ.get("OS_WORK_DIR", os.path.join(ROOT, "work", "opensanctions"))
WORK_CELLS = os.environ.get("OS_WORK_CELLS_DIR", os.path.join(ROOT, "work", "opensanctions_cells"))
COHORT = os.environ.get("OS_COHORT", "cells")
WORK = WORK_CELLS if COHORT == "cells" else WORK_FIRST
_R = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
RESULTS = os.path.join(_R, "opensanctions" if COHORT == "cells" else "opensanctions_first")
