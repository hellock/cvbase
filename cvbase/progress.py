from multiprocessing import Pool
from sys import stdout

from cvbase.timer import Timer


class ProgressBar(object):
    """A progress bar which can print the progress"""

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        self.bar_width = 50
        self.completed = 0
        if start:
            self.start()

    def start(self):
        if self.task_num > 0:
            stdout.write('[{}] 0/{}, ETA:'.format(' ' * self.bar_width,
                                                  self.task_num))
        else:
            stdout.write('completed: 0, elapsed: {:5}s'.format(0))
        stdout.flush()
        self.timer = Timer()

    def update(self):
        self.completed += 1
        elapsed = self.timer.since_start()
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
            stdout.write('\r[{}] {}/{}, {:.1f} tasks/s, ETA: {:5}s'.format(
                bar_chars, self.completed, self.task_num, fps, eta))
        else:
            stdout.write('completed: {}, elapsed: {:5}s, {:.1f} tasks/s'.
                         format(self.completed, elapsed, fps))
        stdout.flush()


def track_progress(func, tasks, bar_width=50, **kwargs):
    prog_bar = ProgressBar(len(tasks), bar_width)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    stdout.write('\n')
    return results


def track_parallel_progress(func,
                            tasks,
                            process_num,
                            bar_width=50,
                            chunksize=1,
                            keep_order=True):
    pool = Pool(process_num)
    prog_bar = ProgressBar(len(tasks), bar_width)
    results = []
    if keep_order:
        for result in pool.imap(func, tasks, chunksize):
            results.append(result)
            prog_bar.update()
    else:
        for result in pool.imap_unordered(func, tasks, chunksize):
            results.append(result)
            prog_bar.update()
    stdout.write('\n')
    return results
