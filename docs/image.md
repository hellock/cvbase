## Image

This module provides some image processing methods.

### Read/Write/Show
To read or write images files, use `read_img` or `write_img`.
```python
import cvbase as cvb

img = cvb.read_img('test.jpg')
img_ = cvb.read_img(img) # nothing will happen, img_ = img
cvb.write_img(img, 'out.jpg')
```

To read images from bytes
```python
import cvbase as cvb

with open('test.jpg', 'rb') as f:
    data = f.read()
img = cvb.img_from_bytes(data)
```

To show an image file or a loaded image
```python
cvb.show_img('tests/data/color.jpg')

for i in range(10):
    img = np.random.randint(256, size=(100, 100, 3), dtype=np.uint8)
    cvb.show_img(img, win_name='test image', wait_time=200)
```

### Resize
There are lots of resize methods. All resize_* methods have a parameter `return_scale`,
if this param is `False`, then the return value is merely the resized image, otherwise
is a tuple (resized_img, scale).
```python
import cvbase as cvb

# resize to a given size
cvb.resize(img, (1000, 600), return_scale=True)
# resize to the same size of another image
cvb.resize_like(img, dst_img, return_scale=False)
# resize by a ratio
cvb.resize_by_ratio(img, 0.5)
# resize so that the max edge no longer than 1000, short edge no longer than 800
# without changing the aspect ratio
cvb.resize_keep_ar(img, 1000, 800)
# resize to the maximum size
cvb.limit_size(img, 400)
```

### Color space conversion
Supported conversion methods:
- bgr2gray
- gray2bgr
- bgr2rgb
- rgb2bgr
- bgr2hsv
- hsv2bgr

```python
import cvbase as cvb

img = cvb.read_img('tests/data/color.jpg')
img1 = cvb.bgr2rgb(img)
img2 = cvb.rgb2gray(img1)
img3 = cvb.bgr2hsv(img)
```

### Crop
Support single/multiple crop.
```python
import cvbase as cvb
import numpy as np

img = cvb.read_img('tests/data/color.jpg')
bboxes = np.array([10, 10, 100, 120])  # x1, y1, x2, y2
patch = cvb.crop_img(img, bboxes)
bboxes = np.array([[10, 10, 100, 120], [0, 0, 50, 50]])
patches = cvb.crop_img(img, bboxes)
```

Resizing cropped patches.
```python
# upsample patches by 1.2x
patches = cvb.crop_img(img, bboxes, scale_ratio=1.2)
```

### Padding
Pad an image to specific size with given values.
```python
import cvbase as cvb

img = cvb.read_img('tests/data/color.jpg')
img = cvb.pad_img(img, (1000, 1200), pad_val=0)
img = cvb.pad_img(img, (1000, 1200), pad_val=[100, 50, 200])
```