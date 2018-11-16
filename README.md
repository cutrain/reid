Reid sub-system

# install
```bash
pip install -r requirements.txt
```

# simple use
```python
import reid
import cv2
# read video
video = reid.get_data('/path/to/your/file')
# generate pictures
pictures = reid.get_picture(video)
picture = pictures.__next__()
# vehicle detection
bboxes = reid.detect_car(picture)
img = reid.draw_boxes(picture, bboxes)

cv2.imshow('pic', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
