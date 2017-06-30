import time

import cvbase as cvb
import pytest


def test_timer_init():
    timer = cvb.Timer(start=False)
    assert not timer.is_running
    timer.start()
    assert timer.is_running
    timer = cvb.Timer()
    assert timer.is_running


def test_timer_run():
    timer = cvb.Timer()
    time.sleep(1)
    assert abs(timer.since_start() - 1) < 1e-2
    time.sleep(1)
    assert abs(timer.since_last_check() - 1) < 1e-2
    assert abs(timer.since_start() - 2) < 1e-2
    timer = cvb.Timer(False)
    with pytest.raises(cvb.TimerError):
        timer.since_start()
    with pytest.raises(cvb.TimerError):
        timer.since_last_check()


def test_timer_context(capsys):
    with cvb.Timer():
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert abs(float(out) - 1) < 1e-2
    with cvb.Timer(print_tmpl='time: {:.1f}s'):
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert out == 'time: 1.0s\n'
