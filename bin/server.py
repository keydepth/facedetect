#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import numpy as np
import io
import os

import inference
import create_graph
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import cv2

app = Flask(__name__)


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/facedetect', methods=['POST'])
def facedetect():
	output = {
		'status': 'NG'
	}

	# get image data from RPi by cv2.imread
	img_file = request.files['img_file']

	# 変なファイル弾き
	if img_file and allowed_file(img_file.filename):
		filename = secure_filename(img_file.filename)
	else:
		return ''' <p>許可されていない拡張子です</p> '''

	# BytesIOで読み込んでOpenCVで扱える型にする
	f = img_file.stream.read()
	bin_data = io.BytesIO(f)
	file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
	img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

	# ニューラルネットによる推論結果を取得 ##############
	# image = cv2.imread(img)
	img = cv2.resize(img, (64, 64))

	#np.save('test_server.npy', image)

	predict_result = inference.predict(img)
	output['status'] = 'OK'
	output['rank'] = predict_result['rank']

	# predict_result をつかったグラフ生成
	#graph_result = create_graph.create(predict_result)

	return jsonify(output)


if __name__ == '__main__':
	app.run(port=50100, debug=False)
