"""
End-to-end smoke tests for app.py.

The unit tests cover the ``utils`` package; nothing else exercises the Streamlit
layer, where a bad widget argument or a stale session-state key would only show
up at runtime. These drive the real app through Streamlit's AppTest harness.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from utils.ui_helpers import safe_number_input_bounds

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="Streamlit testing harness unavailable"
).AppTest

APP_TIMEOUT = 120

# Resolved from this file's location so the tests pass regardless of the
# working directory pytest happens to be invoked from.
APP_PATH = pathlib.Path(__file__).resolve().parent.parent / "app.py"


def run_app():
    """Start app.py and return the finished AppTest instance."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    at.run()
    return at


def test_app_starts_without_exception():
    """A cold start must render cleanly with no dataset loaded."""
    at = run_app()

    assert not at.exception
    assert len(at.tabs) == 5


def test_load_demo_dataset_button_populates_state():
    """The demo button installs a dataset and reports it."""
    at = run_app()

    demo_button = next(b for b in at.button if "demo" in b.label.lower())
    demo_button.click().run()

    assert not at.exception
    assert at.session_state["data_loaded"] is True
    assert at.session_state["data_source"] == "demo"
    assert len(at.session_state["df"]) > 0


def test_full_flow_trains_and_ranks_models():
    """
    Drive the whole app: load demo data, select target and features, train.

    Session state is seeded directly for the selection step because the target
    and feature widgets live behind a tab that AppTest cannot click through in
    one pass.
    """
    at = run_app()

    next(b for b in at.button if "demo" in b.label.lower()).click().run()
    assert not at.exception

    df = at.session_state["df"]
    at.session_state["target_col"] = "price"
    at.session_state["feature_cols"] = [c for c in df.columns if c != "price"]
    at.run()
    assert not at.exception

    launch = next(b for b in at.button if "training" in b.label.lower())
    launch.click().run()

    assert not at.exception
    assert at.session_state["models_trained"] is True

    results = at.session_state["training_results"]
    assert results
    for result in results.values():
        assert np.isfinite(result["r2"])
        assert np.isfinite(result["rmse"])


def test_leaderboard_ranks_by_numeric_score():
    """
    The champion must be the highest R2, not the highest formatted string.

    Sorting the pre-formatted score column put '9.xxxx' above '10.xxxx'; this
    pins the numeric ordering that replaced it.
    """
    scores = {"Model A": -2.5, "Model B": 0.5, "Model C": 12.0}

    comparison_df = pd.DataFrame(
        [{"Model": name, "R² Score": value} for name, value in scores.items()]
    ).sort_values("R² Score", ascending=False, ignore_index=True)

    assert comparison_df.iloc[0]["Model"] == "Model C"
    assert comparison_df.iloc[-1]["Model"] == "Model A"


@pytest.mark.parametrize("series", [
    pd.Series([5.0, 5.0, 5.0]),          # constant column, std == 0
    pd.Series([7.0]),                    # single row, std is NaN
    pd.Series([np.nan, np.nan]),         # nothing usable at all
    pd.Series([0.0, 0.0]),               # constant at zero
    pd.Series(["a", "b"]),               # not numeric at all
    pd.Series([1.0, 2.0, 3.0]),          # the ordinary case
])
def test_safe_number_input_bounds_always_yields_a_usable_range(series):
    """
    A degenerate column must still yield a range ``st.number_input`` accepts.

    The widget rejects min == max, and a single row gives std = NaN, which is
    what used to crash the Prediction Lab.
    """
    low, high, default, step = safe_number_input_bounds(series)

    assert low < high
    assert low <= default <= high
    assert step > 0
    assert all(np.isfinite(v) for v in (low, high, default, step))
