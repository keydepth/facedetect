#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import inference
import cv2
import numpy as np
import glob

in_dir = "./test_data/*.png"
in_img = sorted(glob.glob(in_dir))

for num in range(len(in_img)):
    image = cv2.imread(str(in_img[num]))
    image = cv2.resize(image, (64, 64))

    # np.save('test_inference.npy', image)

    result = inference.predict(image)

    print(str(in_img[num])+"\t"+str(result))

# i = 1
# for r in result['rank']:
# 	print('{} : {} : {}'.format(str(i), r['no'], str(r['accuracy'])))
# 	i += 1
