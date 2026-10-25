"""Hazard H9/H12: every review output carries the fixed intended-use notice and no forbidden judgements."""
from pathlib import Path

import pandas as pd

from gaitrec_rehab.monitoring import changes_from_first, session_table
from gaitrec_rehab.report import FIXED_NOTICE, render
from gaitrec_rehab.synthetic import synthetic_trials

PROTOTYPE = Path(__file__).parents[1] / "prototype" / "template.html"
FORBIDDEN = ["fit for work", "return to work", "malingering", "compensable", "causation is", "faking"]


def test_report_page_carries_notice_and_no_forbidden_terms():
    p = session_table(synthetic_trials(1, "K", [-0.2, -0.1]))
    html = render(p, changes_from_first(p), [], None).lower()
    assert FIXED_NOTICE.lower() in html
    assert not any(term in html for term in FORBIDDEN)


def test_prototype_carries_notice_and_no_forbidden_terms():
    html = PROTOTYPE.read_text(encoding="utf-8").lower()
    assert "research prototype" in html and "not a diagnosis" in html
    assert "work capacity, causation or impairment" in html
    assert not any(term in html for term in FORBIDDEN)
