Reid sub-system

# Requirement
```
gcc-5
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

# prepare
You should put the pretrained model at ./frcnn/model/faster_rcnn.pth

# cache
Whenever you process a video, it will save in reid/dataset, with the same file name prefix.

# *use* (one step)
```python
import sys
sys.path.append('/path/to/the/reid_module')
import reid
import skimage

query_path = 'a.jpg'
video_path = 'a.mp4'
images_iter = reid.reid(query_path, video_path, k=10)
for image in images_iter:
	image = image[:,:,::-1] # or numpy.flip(image, 2)
	cv2.imshow('a', image)
	k=cv2.waitKey(20)
	if k == ord('q'):
		break
cv2.destroyAllwindows()
```
**NOTE** : image is RGB format, remember to change channel if you use cv2 (which is BGR)

# *use* (step by step)
```python
import sys
sys.path.append('/path/to/the/reid_module')
import reid
import skimage

# read video
video = reid.get_data('/path/to/your/file')
# generate images
images = list(reid.get_image(video))
# person detection
bboxes = list(map(reid.detect, images))
# draw bbox for the first image
img = reid.draw_box(images[0], bboxes[0])
# cut images
for i in range(len(bboxes)):
	data_images = reid.cut_image(images[i], bboxes[i])
# feature extraction
features = reid.get_feature(data_images)
# retrieval
indexs = reid.retrieval(features[0], features, k=10) # you can change features[0] into any other feature you got
find_images = []
for index in indexs:
	find_images.append(data_images[index])

```

