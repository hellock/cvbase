from time import time


class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer(object):

    def __init__(self, start=True):
        self._is_running = False
        if start:
            self.start()

    @property
    def is_running(self):
        return self._is_running

    def start(self):
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    def since_start(self):
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time() - self._t_last
        self._t_last = time()
        return dur


_g_timers = {}  # global timers


def check_time(timer_id):
    """Add check points in a single line

    Args:
        timer_id(str): timer identifier
    """
    if timer_id not in _g_timers:
        _g_timers[timer_id] = Timer()
        return 0
    else:
        return _g_timers[timer_id].since_last_check()
