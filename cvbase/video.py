from collections import OrderedDict
from os import path

import cv2

from cvbase.io import check_file_exist, mkdir_or_exist, scandir
from cvbase.opencv import USE_OPENCV3

if USE_OPENCV3:
    from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                     CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                     CAP_PROP_POS_FRAMES, VideoWriter_fourcc)
else:
    from cv2.cv import CV_CAP_PROP_FRAME_WIDTH as CAP_PROP_FRAME_WIDTH
    from cv2.cv import CV_CAP_PROP_FRAME_HEIGHT as CAP_PROP_FRAME_HEIGHT
    from cv2.cv import CV_CAP_PROP_FPS as CAP_PROP_FPS
    from cv2.cv import CV_CAP_PROP_FRAME_COUNT as CAP_PROP_FRAME_COUNT
    from cv2.cv import CV_CAP_PROP_FOURCC as CAP_PROP_FOURCC
    from cv2.cv import CV_CAP_PROP_POS_FRAMES as CAP_PROP_POS_FRAMES
    from cv2.cv import CV_FOURCC as VideoWriter_fourcc


class Cache(object):

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val


class VideoReader(object):

    def __init__(self, filename, cache_capacity=0):
        check_file_exist(filename, 'Video file not found: ' + filename)
        self._vcap = cv2.VideoCapture(filename)
        self._cache = Cache(cache_capacity) if cache_capacity > 0 else None
        self._position = 0
        # get basic info
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = int(round(self._vcap.get(CAP_PROP_FPS)))
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        return self._vcap

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def fps(self):
        return self._fps

    @property
    def frame_cnt(self):
        return self._frame_cnt

    @property
    def fourcc(self):
        return self._fourcc

    @property
    def position(self):
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        pos = self._position + 1
        if self._cache:
            img = self._cache.get(pos)
            if img:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(pos, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position = pos
        return (ret, img)

    def get_frame(self, frame_id):
        if frame_id <= 0 or frame_id > self._frame_cnt:
            raise ValueError('frame_id must be between 1 and frame_cnt')
        if frame_id == self._position + 1:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img:
                self._position = frame_id
                return (True, img)
        self._set_real_position(frame_id - 1)
        ret, img = self._vcap.read()
        if ret:
            self._position += 1
            if self._cache:
                self._cache.put(self._position, img)
        return (ret, img)

    def current_frame(self):
        if self._position == 0:
            return None
        return self._cache.get(self._position)

    def cvt2frames(self,
                   frame_dir,
                   file_start=0,
                   filename_digit=6,
                   ext='jpg',
                   start=0,
                   max_num=0,
                   print_interval=0):
        mkdir_or_exist(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)
        converted = 0
        while converted < task_num:
            ret, img = self.read()
            if not ret:
                break
            file_idx = converted + file_start
            filename = path.join(
                frame_dir,
                '{0:0{1}d}.{2}'.format(file_idx, filename_digit, ext))
            cv2.imwrite(filename, img)
            converted += 1
            if print_interval > 0 and converted % print_interval == 0:
                print(
                    'video2frame progress: {}/{}'.format(converted, task_num))

    def __iter__(self):
        self._set_real_position(0)
        return self

    def next(self):
        ret, img = self.read()
        if ret:
            return img
        else:
            raise StopIteration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()


def frames2video(frame_dir,
                 video_file,
                 fps=30,
                 fourcc='XVID',
                 filename_digit=6,
                 ext='jpg',
                 start=0,
                 end=0):
    """read the frame images from a directory and join them as a video
    """
    if end == 0:
        max_idx = len([name for name in scandir(frame_dir, ext)]) - 1
    else:
        max_idx = end
    first_file = path.join(frame_dir,
                           '{0:0{1}d}.{2}'.format(start, filename_digit, ext))
    check_file_exist(first_file, 'The start frame not found: ' + first_file)
    img = cv2.imread(first_file)
    height, width = img.shape[:2]
    vwriter = cv2.VideoWriter(video_file,
                              VideoWriter_fourcc(*fourcc), fps,
                              (width, height))
    idx = start
    while idx <= max_idx:
        filename = path.join(frame_dir,
                             '{0:0{1}d}.{2}'.format(idx, filename_digit, ext))
        img = cv2.imread(filename)
        vwriter.write(img)
        idx += 1
    vwriter.release()
