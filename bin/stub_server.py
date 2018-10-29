#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import requests
import json
import cv2
import create_csv
# import create_graph


# XLSX_MIMETYPE = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


# image = cv2.imread('./test_data/0.jpg')
# image = cv2.resize(image, (64, 64))

with open('./test_data/00.11.jpg.png', 'rb') as f:
	files = {'img_file': ('./test_data/00.0.jpg.png', f, 'image/png')}

	res = requests.post('http://localhost:50100/facedetect', files=files)

res_data = res.json()
rank = res_data['result']['rank']

i = 0
for r in rank:
	print('{} : {}'.format(r['no'], r['accuracy']))
	i += 1
print(i)

result = res_data['result']
#
# create_graph.create_graph(result)
create_csv.create(result)
