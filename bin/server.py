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

	# get image data from RPi
	image = request.form['image']

	# ニューラルネットによる推論結果を取得 ##############
	predict_result = inference.predict(image)
	# predict_resultのイメージ
	# {'status': 'OK',		# 'OK' or 'NG'
	#  'data': {'date': '20180929_171203', 'list': [0.03684283420443535, 0.03800936043262482, 0.02651863358914852, 0.015608073212206364, 0.024989087134599686, 0.0308738574385643, 0.01939229480922222, 0.026615887880325317, 0.0351942740380764, 0.046588167548179626, 0.09860555082559586, 0.20099063217639923, 0.022021153941750526, 0.09454210102558136, 0.05241763964295387, 0.049437906593084335, 0.05118785798549652, 0.0604083351790905, 0.06975629180669785], 'rank': [{'no': '12ケーキ屋', 'accuracy': 0.20099063217639923}, {'no': '11デンソー社員', 'accuracy': 0.09860555082559586}, {'no': '14テニスプレイヤ', 'accuracy': 0.09454210102558136}, {'no': '19デンソー社員', 'accuracy': 0.06975629180669785}, {'no': '18医師', 'accuracy': 0.0604083351790905}, {'no': '15テニスプレイヤ', 'accuracy': 0.05241763964295387}, {'no': '17デザイナー', 'accuracy': 0.05118785798549652}, {'no': '16ケーキ屋', 'accuracy': 0.049437906593084335}, {'no': '10演劇(役者)', 'accuracy': 0.046588167548179626}, {'no': '2小学校教師', 'accuracy': 0.03800936043262482}, {'no': '1デンソー社員', 'accuracy': 0.03684283420443535}, {'no': '9演劇(役者)', 'accuracy': 0.0351942740380764}, {'no': '6演劇(脚本)', 'accuracy': 0.0308738574385643}, {'no': '8演劇(役者)', 'accuracy': 0.026615887880325317}, {'no': '3高校教師', 'accuracy': 0.02651863358914852}, {'no': '5美容師', 'accuracy': 0.024989087134599686}, {'no': '13デザイナー', 'accuracy': 0.022021153941750526}, {'no': '7演劇(役者)', 'accuracy': 0.01939229480922222}, {'no': '4アナウンサー', 'accuracy': 0.015608073212206364}], 'top': '12ケーキ屋', 'rect': [60, 49, 88, 88]}
	#  }

	# predict_result をつかったグラフ生成
	graph_result = create_graph.create(predict_result)
	
	output = {
		'status': 'NG',
		'rank': [],
		'graph': graph_result
	}

	return jsonify(output)


if __name__ == '__main__':
	app.run(port=50100)
