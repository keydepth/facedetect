#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import requests
import cv2
import glob
import time


# XLSX_MIMETYPE = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


# image = cv2.imread('./test_data/0.jpg')
# image = cv2.resize(image, (64, 64))

in_dir = "./test_data/*.png"
in_img = sorted(glob.glob(in_dir))

for num in range(len(in_img)):
	start = time.time()
	with open(str(in_img[num]), 'rb') as f:
		files = {'img_file': (str(in_img[num]), f, 'image/png')}
		res = requests.post('http://localhost:50100/facedetect', files=files)
	elapsed_time = time.time() - start

	print(str(in_img[num])+"\t"+str(elapsed_time)+"\t"+res.text)
