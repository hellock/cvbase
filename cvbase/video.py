import functools
import os
import subprocess
import tempfile
from collections import OrderedDict
from os import path

import cv2

from cvbase.io import check_file_exist, mkdir_or_exist, scandir
from cvbase.opencv import USE_OPENCV3
from cvbase.progress import track_progress

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
    """Video class with similar usage to a list object.

    This video warpper class provides convenient apis to access frames.
    There exists an issue of OpenCV's VideoCapture class that jumping to a
    certain frame may be inaccurate. It is fixed in this class by checking
    the position after jumping each time.

    Cache is used when decoding videos. So if the same frame is visited for the second time,
    there is no need to decode again if it is stored in the cache.

    :Example:

    >>> import cvbase as cvb
    >>> v = cvb.VideoReader('sample.mp4')
    >>> len(v)  # get the total frame number with `len()`
    120
    >>> for img in v:  # v is iterable
    >>>     cvb.show_img(img)
    >>> v[5]  # get the 6th frame

    """

    def __init__(self, filename, cache_capacity=10):
        check_file_exist(filename, 'Video file not found: ' + filename)
        self._vcap = cv2.VideoCapture(filename)
        assert cache_capacity > 0
        self._cache = Cache(cache_capacity)
        self._position = 0
        # get basic info
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = int(round(self._vcap.get(CAP_PROP_FPS)))
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: raw VideoCapture object"""
        return self._vcap

    @property
    def opened(self):
        """bool: indicate whether the video is opened"""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: width of video frames"""
        return self._width

    @property
    def height(self):
        """int: height of video frames"""
        return self._height

    @property
    def resolution(self):
        """tuple: video resolution (width, height)"""
        return (self._width, self._height)

    @property
    def fps(self):
        """int: fps of the video"""
        return self._fps

    @property
    def frame_cnt(self):
        """int: total frames of the video"""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "four character code" of the video"""
        return self._fourcc

    @property
    def position(self):
        """int: current cursor position, indicating which frame"""
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
        """Read the next frame

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode and return it, put it in the cache.

        Returns:
            ndarray or None: return the frame if successful, otherwise None.
        """
        pos = self._position + 1
        if self._cache:
            img = self._cache.get(pos)
            if img is not None:
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
        return img

    def get_frame(self, frame_id):
        """Get frame by frame id

        Args:
            frame_id(int): id of the expected frame, 1-based index

        Returns:
            ndarray or None: return the frame if successful, otherwise None.
        """
        if frame_id <= 0 or frame_id > self._frame_cnt:
            raise ValueError(
                '"frame_id" must be between 1 and {}'.format(self._frame_cnt))
        if frame_id == self._position + 1:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id
                return img
        self._set_real_position(frame_id - 1)
        ret, img = self._vcap.read()
        if ret:
            self._position += 1
            if self._cache:
                self._cache.put(self._position, img)
        return img

    def current_frame(self):
        """Get the current frame (frame that is just visited)

        Returns:
            ndarray or None: if the video is fresh, return None, otherwise return the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position)

    def cvt2frames(self,
                   frame_dir,
                   file_start=0,
                   filename_tmpl='{:06d}.jpg',
                   start=0,
                   max_num=0,
                   show_progress=True):
        """Convert a video to frame images

        Args:
            frame_dir(str): output directory to store all the frame images
            file_start(int): from which filename starts
            filename_tmpl(str): filename template, with the index as the variable
            start(int): starting frame index
            max_num(int): maximum number of frames to be written
            show_progress(bool): whether to show a progress bar
        """
        mkdir_or_exist(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)

        def write_frame(file_idx):
            img = self.read()
            filename = path.join(frame_dir, filename_tmpl.format(file_idx))
            cv2.imwrite(filename, img)

        if show_progress:
            track_progress(write_frame,
                           range(file_start, file_start + task_num))
        else:
            for i in range(task_num):
                img = self.read()
                if img is None:
                    break
                filename = path.join(frame_dir,
                                     filename_tmpl.format(i + file_start))
                cv2.imwrite(filename, img)

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, i):
        return self.get_frame(i)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()


def frames2video(frame_dir,
                 video_file,
                 fps=30,
                 fourcc='XVID',
                 filename_tmpl='{:06d}.jpg',
                 start=0,
                 end=0,
                 show_progress=True):
    """Read the frame images from a directory and join them as a video

    Args:
        frame_dir(str): frame directory
        video_file(str): output video filename
        fps(int): fps of the output video
        fourcc(str): foutcc of the output video, this should be compatible with the output file type
        filename_tmpl(str): filename template, with the index as the variable
        start(int): starting frame index
        end(int): ending frame index
        show_progress(bool): whether to show a progress bar
    """
    if end == 0:
        ext = filename_tmpl.split('.')[-1]
        end = len([name for name in scandir(frame_dir, ext)])
    first_file = path.join(frame_dir, filename_tmpl.format(start))
    check_file_exist(first_file, 'The start frame not found: ' + first_file)
    img = cv2.imread(first_file)
    height, width = img.shape[:2]
    resolution = (width, height)
    vwriter = cv2.VideoWriter(video_file,
                              VideoWriter_fourcc(*fourcc), fps, resolution)

    def write_frame(file_idx):
        filename = path.join(frame_dir, filename_tmpl.format(file_idx))
        img = cv2.imread(filename)
        vwriter.write(img)

    if show_progress:
        track_progress(write_frame, range(start, end))
    else:
        for i in range(start, end):
            filename = path.join(frame_dir, filename_tmpl.format(i))
            img = cv2.imread(filename)
            vwriter.write(img)
    vwriter.release()


def check_ffmpeg(func):
    """A decorator to check if ffmpeg is installed"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if subprocess.call('which ffmpeg', shell=True) != 0:
            raise RuntimeError('ffmpeg is not installed')
        func(*args, **kwargs)

    return wrapper


@check_ffmpeg
def convert_video(in_file, out_file, pre_options='', **kwargs):
    """Convert a video with ffmpeg

    This provides a general api to ffmpeg, the executed command is::

        ffmpeg -y <pre_options> -i <in_file> <options> <out_file>

    Options(kwargs) are mapped to ffmpeg commands by the following rules:

    - key=val: "-key val"
    - key=True: "-key"
    - key=False: ""

    Args:
        in_file(str): input video filename
        out_file(str): output video filename
        pre_options(str): options appears before "-i <in_file>"
    """
    options = []
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                options.append('-{}'.format(k))
        elif k == 'log_level':
            assert v in [
                'quiet', 'panic', 'fatal', 'error', 'warning', 'info',
                'verbose', 'debug', 'trace'
            ]
            options.append('-loglevel {}'.format(v))
        else:
            options.append('-{} {}'.format(k, v))
    cmd = 'ffmpeg -y {} -i {} {} {}'.format(pre_options, in_file,
                                            ' '.join(options), out_file)
    print(cmd)
    subprocess.call(cmd, shell=True)


@check_ffmpeg
def resize_video(in_file,
                 out_file,
                 size=None,
                 ratio=None,
                 keep_ar=False,
                 log_level='info',
                 **kwargs):
    """Resize a video

    Args:
        in_file(str): input video filename
        out_file(str): output video filename
        size(tuple): expected (w, h), eg, (320, 240) or (320, -1)
        ratio(tuple or float): expected resize ratio, (2, 0.5) means (w*2, h*0.5)
        keep_ar(bool): whether to keep original aspect ratio
        log_level(str): log level of ffmpeg
    """
    if size is None and ratio is None:
        raise ValueError('expected size or ratio must be specified')
    elif size is not None and ratio is not None:
        raise ValueError('size and ratio cannot be specified at the same time')
    options = {'log_level': log_level}
    if size:
        if not keep_ar:
            options['vf'] = 'scale={}:{}'.format(size[0], size[1])
        else:
            options['vf'] = ('scale=w={}:h={}:force_original_aspect_ratio'
                             '=decrease'.format(size[0], size[1]))
    else:
        if not isinstance(ratio, tuple):
            ratio = (ratio, ratio)
        options['vf'] = 'scale="trunc(iw*{}):trunc(ih*{})"'.format(
            ratio[0], ratio[1])
    convert_video(in_file, out_file, **options)


@check_ffmpeg
def cut_video(in_file,
              out_file,
              start=None,
              end=None,
              vcodec=None,
              acodec=None,
              log_level='info',
              **kwargs):
    """Cut a clip from a video

    Args:
        in_file(str): input video filename
        out_file(str): output video filename
        start(None or float): start time (in seconds)
        end(None or float): end time (in seconds)
        vcodec(None or str): output video codec, None for unchanged
        acodec(None or str): output audio codec, None for unchanged
        log_level(str): log level of ffmpeg
    """
    options = {'log_level': log_level}
    if vcodec is None:
        options['vcodec'] = 'copy'
    if acodec is None:
        options['acodec'] = 'copy'
    if start:
        options['ss'] = start
    else:
        start = 0
    if end:
        options['t'] = end - start
    convert_video(in_file, out_file, **options)


@check_ffmpeg
def concat_video(video_list,
                 out_file,
                 vcodec=None,
                 acodec=None,
                 log_level='info',
                 **kwargs):
    """Concatenate multiple videos into a single one

    Args:
        video_list(list): a list of video filenames
        out_file(str): output video filename
        vcodec(None or str): output video codec, None for unchanged
        acodec(None or str): output audio codec, None for unchanged
        log_level(str): log level of ffmpeg
    """
    _, tmp_filename = tempfile.mkstemp(suffix='.txt', text=True)
    with open(tmp_filename, 'w') as f:
        for filename in video_list:
            f.write('file {}\n'.format(path.abspath(filename)))
    options = {'log_level': log_level}
    if vcodec is None:
        options['vcodec'] = 'copy'
    if acodec is None:
        options['acodec'] = 'copy'
    convert_video(
        tmp_filename, out_file, pre_options='-f concat -safe 0', **options)
    os.remove(tmp_filename)
