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
