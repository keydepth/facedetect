#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import inference
import cv2
import numpy as np


image = cv2.imread('./test_data/00.0.jpg')
image = cv2.resize(image, (64, 64))

# np.save('test_inference.npy', image)

result = inference.predict(image)

rank = result['rank']

i = 0
for r in rank:
	i += 1
	print('{} : {}'.format(r['no'], r['accuracy']))

print(i)
# i = 1
# for r in result['rank']:
# 	print('{} : {} : {}'.format(str(i), r['no'], str(r['accuracy'])))
# 	i += 1
