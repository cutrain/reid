Reid sub-system

# Requirement
```
gcc version <= 6
cuda9.0
```

# install
```bash
git clone https://github.com/cutrain/reid
cd reid
pip install -r requirements.txt
cd frcnn/lib
./make.sh
```

# *use* (step by step)
```python
import sys
sys.path.append('/path/to/the/reid_module')
import reid
import skimage

# read video
video = reid.get_data('/path/to/your/file')
# generate pictures
pictures = list(reid.get_picture(video))
# vehicle detection
bboxes = reid.detect_car(pictures)
img = reid.draw_box(pictures[0], bboxes[0])
skimage.imshow(img)
# person detection
person_bboxes = reid.detect(pictures)
img = reid.draw_box(pictures[0], bboxes[0])
skimage.imshow(img)

```

# *use* (recognize persons from a list of video)
```python
import sys
sys.path.append('/path/to/the/reid_module')
import reid
import skimage

person_paths = ['a.jpg', 'b.jpg']
video_paths = ['a.mp4', 'b.mp4']
imgs = reid.person_reid(person_paths, video_paths, k=10)
for i in range(len(person_paths)):
	for person in imgs[i]:
		skimage.io.imshow(person)
```

