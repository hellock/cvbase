from os import path
from cvbase import VideoReader


def test_read():
    video_path = path.join(path.dirname(__file__), 'data/test.mp4')
    video = VideoReader(video_path)
    assert video.width == 294
    assert video.height == 240
    assert video.fps == 25
    assert video.frame_cnt == 168
    assert video.position == 0