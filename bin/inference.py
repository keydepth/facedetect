#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import cv2
from keras.models import load_model
from datetime import datetime
import socket
import json
import websocket
from websocket import create_connection
import csv

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import io
import os
from keras import backend as K
import copy

app = Flask(__name__)
model = None

tcpsend = False
address = ('localhost', 12345)
max_size = 10000

websocketsend = False
websocketaddress = "ws://localhost:6789/"

logfile = './log/log.csv'
targetpath = './target_image/'
targetexp = 'png'
# h5File='./bin/my_model-epoch20.h5'
# h5File='./bin/my_model-n19-epoch25.h5'
# h5File='./bin/my_model-n44-epoch17.h5'
# h5File='./bin/my_model-n56-epoch17.h5'
h5File = './bin/my_model-n59-epoch17.h5'
facedetect = './bin/haarcascade_frontalface_alt.xml'

# モデル読み込み
# global model
# model = load_model(h5File)


# 出力重み
weights = []

# job
jobs = []

# グラフ表示ラベル
labels = ['1デンソー社員', '2小学校教師', '3高校教師', '4アナウンサー', '5美容師', '6演劇(脚本)', '7演劇(役者)', '8演劇(役者)', '9演劇(役者)', '10演劇(役者)',
          '11デンソー社員', '12ケーキ屋', '13デザイナー', '14テニスプレイヤ', '15テニスプレイヤ', '16ケーキ屋', '17デザイナー', '18医師', '19デンソー社員', '20看護師',
          '21臨床検査', '22臨床検査', '23医師', '24看護師', '25看護師', '26放射線技師', '27薬剤師', '28臨床検査技師', '29放射線技師', '30臨床検査技師', '31薬剤師',
          '32看護師', '33放射線技師', '34放射線技師', '35薬剤師', '36放射線技師', '37臨床検査技師', '38看護師', '39看護師', '40看護師', '41看護師', '42臨床検査技師',
          '43社長', '44警察官', '45トヨタ社員', '46アイシン社員', '47アイシン社員', '48政治家', '49プロ野球選手', '50プロ野球選手', '51プロ野球選手', '52プロサッカー選手',
          '53プロサッカー選手', '54プロサッカー選手', '55ユーチューバー', '56ユーチューバー', '57トヨタ社員', '58ゲームクリエータ', '59ケーキ屋']

# 学習データのパラメータテーブル
# 独自性, 有名度, 財力
matTable = np.matrix([
	[0.5, 0.5, 0.5],  # 1デンソー社員
	[0.5, 0.5, 0.5],  # 2小学校教師
	[0.5, 0.5, 0.5],  # 3高校教師
	[0.5, 0.5, 0.5],  # 4アナウンサー
	[0.5, 0.5, 0.5],  # 5美容師
	[1.0, 0.5, 0.5],  # 6演劇(脚本)
	[0.5, 0.5, 0.5],  # 7演劇(役者)
	[0.5, 0.5, 0.5],  # 8演劇(役者)
	[0.5, 0.5, 0.5],  # 9演劇(役者)
	[0.5, 0.5, 0.5],  # 10演劇(役者)
	[0.5, 0.5, 0.5],  # 11デンソー社員
	[0.5, 0.5, 0.5],  # 12ケーキ屋
	[0.5, 0.5, 0.5],  # 13デザイナー
	[1.0, 1.0, 1.0],  # 14テニスプレイヤ
	[1.0, 1.0, 1.0],  # 15テニスプレイヤ
	[0.5, 0.5, 0.5],  # 16ケーキ屋
	[0.5, 0.5, 0.5],  # 17デザイナー
	[0.5, 0.5, 0.5],  # 18医師
	[0.5, 0.5, 0.5],  # 19デンソー社員
	[0.5, 0.5, 0.8],  # 20看護師
	[0.5, 0.5, 0.8],  # 21臨床検査
	[0.5, 0.5, 0.8],  # 22臨床検査
	[0.5, 0.5, 0.9],  # 23医師
	[0.5, 0.5, 0.8],  # 24看護師
	[0.5, 0.5, 0.8],  # 25看護師
	[0.5, 0.5, 0.8],  # 26放射線技師
	[0.5, 0.5, 0.8],  # 27薬剤師
	[0.5, 0.5, 0.8],  # 28臨床検査技師
	[0.5, 0.5, 0.8],  # 29放射線技師
	[0.5, 0.5, 0.8],  # 30臨床検査技師
	[0.5, 0.5, 0.8],  # 31薬剤師
	[0.5, 0.5, 0.8],  # 32看護師
	[0.5, 0.5, 0.8],  # 33放射線技師
	[0.5, 0.5, 0.8],  # 34放射線技師
	[0.5, 0.5, 0.8],  # 35薬剤師
	[0.5, 0.5, 0.8],  # 36放射線技師
	[0.5, 0.5, 0.8],  # 37臨床検査技師
	[0.5, 0.5, 0.8],  # 38看護師
	[0.5, 0.5, 0.8],  # 39看護師
	[0.5, 0.5, 0.8],  # 40看護師
	[0.5, 0.5, 0.8],  # 41看護師
	[0.5, 0.5, 0.8],  # 42臨床検査技師
	[1.0, 0.5, 1.0],  # 43社長
	[0.5, 0.5, 0.5],  # 44警察官
	[0.5, 0.5, 0.6],  # 45トヨタ社員
	[0.5, 0.5, 0.5],  # 46アイシン社員
	[0.5, 0.5, 0.5],  # 47アイシン社員
	[0.8, 0.8, 0.7],  # 48政治家
	[1.0, 1.0, 1.0],  # 49プロ野球選手
	[1.0, 1.0, 1.0],  # 50プロ野球選手
	[1.0, 1.0, 1.0],  # 51プロ野球選手
	[1.0, 1.0, 1.0],  # 52プロサッカー選手
	[1.0, 1.0, 1.0],  # 53プロサッカー選手
	[1.0, 1.0, 1.0],  # 54プロサッカー選手
	[1.0, 1.0, 1.0],  # 55ユーチューバー
	[1.0, 1.0, 1.0],  # 56ユーチューバー
	[1.0, 1.0, 1.0],  # 57トヨタ社員
	[1.0, 1.0, 1.0],  # 56ゲームクリエータ
	[1.0, 1.0, 1.0]  # 56ケーキ屋

])

csvmatrixfile = './bin/matrix.csv'

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)


def allowed_file(filename):
	return '.' in filename and \
	       filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# Matrix(csv)の読み込み
def loadMatrix():
	with open(csvmatrixfile, 'r') as f:
		reader = csv.reader(f)
		header = next(reader)  # ヘッダーを読み飛ばしたい時
		i = 0
		for row in reader:
			#			print(row)          # 1行づつ取得できる
			matTable[i, 0] = float(row[0])
			matTable[i, 1] = float(row[1])
			matTable[i, 2] = float(row[2])
			weights.append(float(row[3]))
			labels[i] = row[4].strip()
			jobs.append(row[5].strip())
			i += 1


# print(matTable)
# print(labels)
# マトリックスの読み込み
loadMatrix()


# print(matTable)
# print(labels)
# print(weights)
# print(jobs)

def detect_face(image, imageOrg, detect):
	#    print(image.shape)
	# opencvを使って顔抽出
	image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cascade = cv2.CascadeClassifier(facedetect)
	# 顔認識の実行
	face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))
	# 顔が１つ以上検出された時
	detecf_exec = False
	if len(face_list) > 0:
		for rect in face_list:
			x, y, width, height = rect
			#            print(rect)
			cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (255, 0, 0), thickness=3)
			img = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
			if img.shape[0] < 64:
				#                print("too small")
				continue
			img = cv2.resize(img, (64, 64))
			img = np.expand_dims(img, axis=0)
			name = ''
			if detect == True:
				name = detect_who(img, imageOrg, int(x), int(y), int(width), int(height))['top']
				detecf_exec = True
			cv2.putText(image, name, (x, y + height + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
	# 顔が検出されなかった時
	#    else:
	#        print("no face")

	# 検出実行時に顔検出されなかった場合、画像全体を使用する。
	if detecf_exec == False:
		if detect == True:
			height, width, channels = image.shape[:3]
			rect = [0, 0, width, height]
			#            print(rect)
			cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[2:4]), (255, 0, 0), thickness=3)
			img = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
			img = cv2.resize(img, (64, 64))
			img = np.expand_dims(img, axis=0)
			name = detect_who(img, imageOrg, rect[0], rect[1], rect[2], rect[3])['top']
	return image


def get_rank_index(my_array):
	# 上位件数
	K = 3
	# ソートはされていない上位k件のインデックス
	unsorted_max_indices = np.argpartition(-my_array, K)[:K]
	# 上位k件の値
	y = my_array[unsorted_max_indices]
	# 大きい順にソートし、インデックスを取得
	indices = np.argsort(-y)
	# 類似度上位k件のインデックス
	max_k_indices = unsorted_max_indices[indices]
	return max_k_indices


def detect_who(img, image, x, y, w, h):
	#    print([x,y,w,h])
	model = load_model(h5File)

	logdate = datetime.now()
	strDate = logdate.strftime("%Y%m%d_%H%M%S")
	ImageFile = targetpath + strDate + '.' + targetexp
	#    print('#### image output = .'+ImageFile)
	cv2.imwrite(ImageFile, image)
	# 予測
	name = ""
	# K.clear_session()
	nameNumLabel = model.predict(img)[0]
	# 上位のindexを取得する
	rank_index = get_rank_index(nameNumLabel)

	matRecog = copy.deepcopy(nameNumLabel)
	i = 0
	for w in weights:
		if i in rank_index:
			matRecog[i] *= w
		else:
			matRecog[i] *= 0
		i += 1
	# 正規化(sum(matRecog)=1)
	matRecog /= sum(matRecog)
	#    print(matRecog)
	dream = np.dot(matRecog, matTable).tolist()[0]
	#    print(dream.tolist())

	result = {}
	result['date'] = strDate
	result['list'] = nameNumLabel.tolist()
	list_dict = []
	n = 0
	for i in result['list']:
		list_dict.append({'no': labels[n], 'accuracy': i, 'job': jobs[n]})
		n += 1
	result['rank'] = sorted(list_dict, key=lambda x: x['accuracy'], reverse=True)
	result['top'] = result['rank'][0]['no']
	result['dream'] = dream
	result['rect'] = [x, y, w, h]
	#    print(result)
	with open(logfile, 'a') as f:
		writer = csv.writer(f)
		f.write(logdate.strftime("%Y/%m/%d %H:%M:%S, "))
		f.write(ImageFile + ', ')
		f.write(str(x) + ', ' + str(y) + ', ' + str(w) + ', ' + str(h) + ', ')
		writer.writerow(nameNumLabel.tolist() + dream)
	if tcpsend == True:
		#        print('Starting the client at', datetime.now())
		client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		client.connect(address)
		client.sendall(json.dumps(result).encode('utf-8'))
		data = client.recv(max_size)
		#        print('At', datetime.now(), 'someone replied', data)
		client.close()
	if websocketsend == True:
		#        print("create websock")
		ws = create_connection(websocketaddress)
		#        print("Sending 'Hello, World'...")
		ws.send(json.dumps({"type": "recog", "data": result}).encode('utf-8'))
		#        rc =  ws.recv()
		#        print("Received '%s'" % rc)
		ws.close()
	return result


# image:imreadした画像
def predict(imageOrg):
	b, g, r = cv2.split(imageOrg)
	imageRGB = cv2.merge([r, g, b])
	imageExpand = np.expand_dims(imageRGB, axis=0)
	return detect_who(imageExpand, imageOrg, 0, 0, 64, 64)


@app.route('/facedetect', methods=['POST'])
def facedetect():
	output = {
		'status': 'NG'
	}
	K.clear_session()
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
	img = cv2.resize(img, (64, 64))

	# ニューラルネットによる推論結果を取得 ##############
	predict_result = predict(img)
	output['status'] = 'OK'
	output['result'] = predict_result

	return jsonify(output)


if __name__ == '__main__':  # pyを実行すると以下が実行される（モジュールとして読み込んだ場合は実行されない）
	# app.run(port=50100, debug=False)
	app.run(host='0.0.0.0', port=50100, debug=False)
