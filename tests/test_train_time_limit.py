"""tests/test_train_time_limit.py: the wall-clock guard that keeps a Kaggle run inside its time limit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from train import out_of_time  # noqa: E402


def test_no_limit_never_stops():
    assert not out_of_time(0.0, None, [9999.0], now=1e9)


def test_first_epoch_starts_while_budget_remains():
    assert not out_of_time(0.0, 10, [], now=60.0)
    assert out_of_time(0.0, 10, [], now=601.0)


def test_stops_when_next_epoch_would_overrun():
    # 30 min budget, 20 min used, epochs take 9.5 min (x1.1 = 10.45 > 10 left)
    assert out_of_time(0.0, 30, [570.0], now=1200.0)
    assert not out_of_time(0.0, 30, [500.0], now=1200.0)
