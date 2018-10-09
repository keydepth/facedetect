#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import inference
import cv2


image = cv2.imread('./test_data/0.jpg')
image = cv2.resize(image, (64, 64))

result = inference.predict(image)

print(result)

# i = 1
# for r in result['rank']:
# 	print('{} : {} : {}'.format(str(i), r['no'], str(r['accuracy'])))
# 	i += 1
