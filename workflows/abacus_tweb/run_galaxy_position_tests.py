#!/usr/bin/env python3
"""Run the focused annotation and affected P10 regression suite with stdlib unittest."""
from pathlib import Path
import os
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
loader=unittest.TestLoader();suite=unittest.TestSuite()
for folder,pattern in [('tests','test_galaxy_position_annotation.py'),
                       ('tests','test_galaxy_label_equivalence.py'),
                       ('tests','test_product_lifecycle.py'),
                       ('tests','test_galaxy_truth_consumers.py'),
                       ('tests/phase4','test_p10_phase_index.py'),
                       ('tests/phase4','test_p10_phase_products.py'),
                       ('tests/phase4','test_p10_bright_parent.py'),
                       ('tests/phase4','test_p10_observed_truth.py')]:
    suite.addTests(unittest.TestLoader().discover(folder,pattern=pattern))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
