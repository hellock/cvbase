import time

from cvbase.progress import ProgressBar


class TestProgressBar(object):

    def test_start(self, capsys):
        # without total task num
        prog_bar = ProgressBar()
        out, _ = capsys.readouterr()
        assert out == 'completed: 0, elapsed: 0s'
        prog_bar = ProgressBar(start=False)
        out, _ = capsys.readouterr()
        assert out == ''
        prog_bar.start()
        out, _ = capsys.readouterr()
        assert out == 'completed: 0, elapsed: 0s'
        # with total task num
        prog_bar = ProgressBar(10)
        out, _ = capsys.readouterr()
        assert out == '[{}] 0/10, elapsed: 0s, ETA:'.format(' ' * 50)
        prog_bar = ProgressBar(10, start=False)
        out, _ = capsys.readouterr()
        assert out == ''
        prog_bar.start()
        out, _ = capsys.readouterr()
        assert out == '[{}] 0/10, elapsed: 0s, ETA:'.format(' ' * 50)

    def test_update(self, capsys):
        # without total task num
        prog_bar = ProgressBar()
        capsys.readouterr()
        time.sleep(1)
        prog_bar.update()
        out, _ = capsys.readouterr()
        assert out == 'completed: 1, elapsed: 1s, 1.0 tasks/s'
        # with total task num
        prog_bar = ProgressBar(10)
        capsys.readouterr()
        time.sleep(1)
        prog_bar.update()
        out, _ = capsys.readouterr()
        assert out == ('\r[{}] 1/10, 1.0 task/s, elapsed: 1s, ETA:     9s'.
                       format('>' * 5 + ' ' * 45))
