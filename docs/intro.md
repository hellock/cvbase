# Introduction

`cvbase` is a miscellaneous set of tools which maybe helpful for computer vision research.
It comprises the following parts.

## Popular features
There are some popular features such as progress visualization, timer, video to frames/frames to videos.


- Progress visualization

    If you want to apply a method to a list of items and track the progress, `track_progress`
    is a good choice. It will display a progress bar to tell the progress and ETA.

    ```python
    import cvbase as cvb

    def func(item):
        # do something
        pass

    tasks = [item_1, item_2, ..., item_n]

    cvb.track_progress(func, tasks)
    ```

    The output is like the following.
    ![progress](_static/progress.gif)

    There is another method `track_parallel_progress`, which wraps multiprocessing and
    progress visualization.

    ```python
    import cvbase as cvb

    def func(item):
        # do something
        pass

    tasks = [item_1, item_2, ..., item_n]

    cvb.track_progress(func, tasks, 8)
    # 8 workers
    ```

- Timer

    It is convinient to computer the runtime of a code block with `Timer`.

    ```python
    import time
    import cvbase as cvb

    with cvb.Timer():
        # there can be any code block
        time.sleep(1)
    ```

    Or try a more flexible way.

    ```python
    import cvbase as cvb

    timer = cvb.Timer()
    # code block 1 here
    print(timer.since_start())
    # code block 2 here
    print(timer.since_last_check())
    print(timer.since_start())
    ```

- Video/Frames conversion

    To split a video into frames.

    ```python
    import cvbase as cvb

    video = cvb.VideoReader('video_file.mp4')
    video.cvt2frames('frame_dir')
    ```
    Besides `cvt2frames`, `VideoReader` wraps many other useful methods to operate a video.

    To generate a video from frames, use the `frames2video` method.

    ```python
    import cvbase as cvb

    video = cvb.VideoReader('frame_dir', 'out_video_file.avi', fps=30)
    ```