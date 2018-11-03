#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import inference
import cv2
import numpy as np
import pickle
import glob
from matplotlib import pyplot as plt


data_list = []

path_list = glob.glob('./test_data/miss_universe/miss universe Amelia Vega/*.png')

# print(path_list)
i = 0
for path in path_list[:10]:
	i += 1
	image = cv2.imread(path)
	image = cv2.resize(image, (64, 64))

	# np.save('test_inference.npy', image)
	print('======start : {}  {}==='.format(str(i), path))
	print()

	result = inference.predict(image)
	data_list.append(result)

with open('data_list.pickle', 'wb') as f:
	pickle.dump(data_list, f)

print(i)

# plt.bar(range(len(data_list)), data_list)
# plt.savefig('predict_data.png')


# i = 0
# for r in rank:
# 	i += 1
# 	print('{} : {}'.format(r['no'], r['accuracy']))
#
# print(i)

