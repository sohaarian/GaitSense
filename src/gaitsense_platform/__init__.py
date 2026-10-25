"""GaitSense v3.0 research platform: two model families on one data foundation.

Capability 1 (biomechanical forecasting and monitoring) wraps the frozen, validated v2.0 system in
`gaitsense_ai` and can run on its own. Capability 2 (clinical recovery prediction) is a separate model family that
runs only on a verified dataset of independently measured clinical outcomes (see `gate.py`), and asks whether
Capability 1 forecasts add predictive value beyond clinical data alone.

Nothing here makes benefit, impairment, MMI or work-capacity decisions.
"""
__version__ = "3.0.0"
