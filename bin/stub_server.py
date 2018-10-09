#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import requests
import cv2


image = cv2.imread('./test_data/0.jpg')
image = cv2.resize(image, (64, 64))

data = {
	'image': image
}

res = requests.post('http://localhost:50100/facedetect', data)

print(res.text)
