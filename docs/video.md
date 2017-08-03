## Video

This module provides friendly apis to read videos.

```python
import cvbase as cvb

video = cvb.VideoReader('test.mp4')
# access basic info
print(video.width, video.height, video.fps, video.frame_cnt)
# iterate over all frames
for frame in video:
    print(frame.shape)
# read the next frame
ret, img = video.read()
# read a frame by index
ret, img = video.get_frame(100)
# split a video into frames and save to a folder
video.cvt2frames('out_dir')
# generate video from frames
cvb.test_frames2video('out_dir', 'test.avi')
```