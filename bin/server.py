#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import inference
import create_graph
from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/facedetect', methods=['POST'])
def facedetect():
	output = {
		'status': 'NG'
	}

	# get image data from RPi by cv2.imread
	image = request.form['image']

	# ニューラルネットによる推論結果を取得 ##############
	predict_result = inference.predict(image)
	output['status'] = 'OK'
	output['rank'] = predict_result['rank']

	# predict_result をつかったグラフ生成
	#graph_result = create_graph.create(predict_result)

	return jsonify(output)


if __name__ == '__main__':
	app.run(port=50100, debug=True)
