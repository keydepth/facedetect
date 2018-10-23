#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import requests
import cv2


# XLSX_MIMETYPE = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


# image = cv2.imread('./test_data/0.jpg')
# image = cv2.resize(image, (64, 64))

with open('./test_data/0.jpg', 'rb') as f:
	files = {'img_file': ('test_image.jpg', f, 'image/jpeg')}

	res = requests.post('http://localhost:50100/facedetect', files=files)

print(res.text)
