Reid sub-system

# install
```bash
pip install -r requirements.txt
```

# simple use
```python
import reid
import cv2
video = reid.get_data('/path/to/your/file')
pictures = reid.get_picture(video)
for picture in pictures:
	cv2.imshow('pic', picture)
	k = cv2.waitKey(30) & 0xff
	if k == 29:
		break
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
