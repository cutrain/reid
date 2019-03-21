Reid sub-system

# Requirement
```
python3.6
GPU Memory >= 6G
pytorch 0.4.0
```

# install
```bash
git clone https://github.com/cutrain/reid
cd reid
#python3 -m venv venv
#source venv/bin/activate
pip install -r requirements.txt
cd yolov3
wget https://pjreddie.com/media/files/yolov3.weights 
cd ..
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

# prepare
you need a pre_trained model 'reid/preid/net.pth', contact me pls

# cache
Whenever you process a video, it will save in reid/dataset, with the same file name prefix.

# find nearby person
```python
import sys
sys.path.append('/path/to/the/reid_module')
import reid

query_path = 'data/a.jpg'
video_path = 'data/a.mp4'
nearby_person = reid.nearby(query_path, video_path, threshold=0.95)
query_image = cv2.imread(query_person)[:,:,::-1]
query_person = reid.cut_image(query_image, reid.detect(query_image)[0])[0]
images_iter = reid.reid([query_path, nearby_person], video_path, threshold=0.95, optimize_query=False)
for image in images_iter:
	cv2.imshow('a', image)
	k = cv2.waitKey(20)
	if k == ord('q'):
		break
cv2.destroyAllWindows()
```

# *auto mark* ***(HERE!!)***
```python
import reid
path = './a.avi'
result = reid.auto_mark(path, threshold=0.99)
```
## result format
```json
{
  'video_name':'./a.avi',
  'result':[
    {
      'object_num':1,
      'coordinate_matrix':[[10,245,20,144]], # using 'numpy' like "image[10:245,20:144]"
      'id':[1]
    },
  ]
}
```


# *use* (one step)
```python
import cv2
import sys
sys.path.append('/path/to/the/reid_module')
import reid

query_path = 'data/a.jpg'
video_path = 'data/a.mp4'
images_iter = reid.reid(query_path, video_path, exist_object=True, threshold=0.95, start_frame=500, frame_count=1000, optimize_query=False)
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
indexs = reid.evaluate(features[0], features, threshold=0.95) # you can change features[0] into any other feature you got
find_images = []
for index in indexs:
	find_images.append(data_images[index])

```

