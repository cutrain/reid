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

#faster rcnn for reid人物切割功能
```
1：设置坐标json存储路径 in  demo.py line 407
2:
```
```python
import ./lib/data/save_json.py 
image = cut_image(path) # 读取存储目标信息文件，裁剪图片
```
