## Image

This module provides some image processing methods.

### Read/Write
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