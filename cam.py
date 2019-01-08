import cv2
import time
import pyfakewebcam
import numpy as np
from queue import Queue

camera = None
cam_size = None
def init_cam(width=640, height=480):
    global camera, cam_size
    cam_size = (width, height)
    print('init -> ', width, height)
    camera = pyfakewebcam.FakeWebcam('/dev/video0', width, height)

def sendweb_reid(image):
    global camera, cam_size
    if image.shape[0] != cam_size[1] or image.shape[1] != cam_size[0]:
        image = cv2.resize(image, cam_size)
    camera.schedule_frame(image)

def webcam(data_queue, width, height, syn):
    pre = time.time() - 1
    init_cam(width=width, height=height)
    while True:
        frame = data_queue.get()
        now = time.time()
        gap = now - pre
        iswait = True
        while syn and gap > 1./25 and not data_queue.empty():
            gap -= 1./25
            frame = data_queue.get()
            iswait = False
        wait = min(1./25, max(0.001, (1./25-gap)))
        if iswait:
            time.sleep(wait)
        pre = time.time()
        sendweb_reid(frame)
        print('gap:', gap, 'wait:', wait)

