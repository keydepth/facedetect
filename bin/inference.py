import random
from keras.utils.np_utils import to_categorical
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from imutils.video import VideoStream
from imutils.video import FPS
import sys
import csv
from datetime import datetime
import socket
import json
import websocket
from websocket import create_connection

tcpsend=False
address = ('localhost', 12345)
max_size = 10000

websocketsend=True
websocketaddress = "ws://localhost:6789/"

labels=['1デンソー社員','2小学校教師','3高校教師','4アナウンサー','5美容師','6演劇(脚本)','7演劇(役者)','8演劇(役者)','9演劇(役者)','10演劇(役者)','11デンソー社員','12ケーキ屋','13デザイナー','14テニスプレイヤ','15テニスプレイヤ','16ケーキ屋','17デザイナー','18医師','19デンソー社員','20看護師','21臨床検査','22臨床検査','23医師','24看護師','25看護師','26放射線技師','27薬剤師','28臨床検査技師','29放射線技師','30臨床検査技師','31薬剤師','32看護師','33放射線技師','34放射線技師','35薬剤師','36放射線技師','37臨床検査技師','38看護師','39看護師','40看護師','41看護師','42臨床検査技師','43社長','44警察官','45トヨタ社員','46アイシン社員','47アイシン社員','48政治家','49プロ野球選手','50プロ野球選手','51プロ野球選手','52プロサッカー選手','53プロサッカー選手','54プロサッカー選手','55ユーチューバー','56ユーチューバー']

# 学習データのパラメータテーブル
# 独自性, 有名度, 財力
matTable = np.matrix([
    [0.5, 0.5, 0.5],   # 1デンソー社員
    [0.5, 0.5, 0.5],   # 2小学校教師
    [0.5, 0.5, 0.5],   # 3高校教師
    [0.5, 0.5, 0.5],   # 4アナウンサー
    [0.5, 0.5, 0.5],   # 5美容師
    [1.0, 0.5, 0.5],   # 6演劇(脚本)
    [0.5, 0.5, 0.5],   # 7演劇(役者)
    [0.5, 0.5, 0.5],   # 8演劇(役者)
    [0.5, 0.5, 0.5],   # 9演劇(役者)
    [0.5, 0.5, 0.5],   # 10演劇(役者)
    [0.5, 0.5, 0.5],   # 11デンソー社員
    [0.5, 0.5, 0.5],   # 12ケーキ屋
    [0.5, 0.5, 0.5],   # 13デザイナー
    [1.0, 1.0, 1.0],   # 14テニスプレイヤ
    [1.0, 1.0, 1.0],   # 15テニスプレイヤ
    [0.5, 0.5, 0.5],   # 16ケーキ屋
    [0.5, 0.5, 0.5],   # 17デザイナー
    [0.5, 0.5, 0.5],   # 18医師
    [0.5, 0.5, 0.5],   # 19デンソー社員
    [0.5, 0.5, 0.8],   # 20看護師
    [0.5, 0.5, 0.8],   # 21臨床検査
    [0.5, 0.5, 0.8],   # 22臨床検査
    [0.5, 0.5, 0.9],   # 23医師
    [0.5, 0.5, 0.8],   # 24看護師
    [0.5, 0.5, 0.8],   # 25看護師
    [0.5, 0.5, 0.8],   # 26放射線技師
    [0.5, 0.5, 0.8],   # 27薬剤師
    [0.5, 0.5, 0.8],   # 28臨床検査技師
    [0.5, 0.5, 0.8],   # 29放射線技師
    [0.5, 0.5, 0.8],   # 30臨床検査技師
    [0.5, 0.5, 0.8],   # 31薬剤師
    [0.5, 0.5, 0.8],   # 32看護師
    [0.5, 0.5, 0.8],   # 33放射線技師
    [0.5, 0.5, 0.8],   # 34放射線技師
    [0.5, 0.5, 0.8],   # 35薬剤師
    [0.5, 0.5, 0.8],   # 36放射線技師
    [0.5, 0.5, 0.8],   # 37臨床検査技師
    [0.5, 0.5, 0.8],   # 38看護師
    [0.5, 0.5, 0.8],   # 39看護師
    [0.5, 0.5, 0.8],   # 40看護師
    [0.5, 0.5, 0.8],   # 41看護師
    [0.5, 0.5, 0.8],   # 42臨床検査技師
    [1.0, 0.5, 1.0],   # 43社長
    [0.5, 0.5, 0.5],   # 44警察官
    [0.5, 0.5, 0.6],   # 45トヨタ社員
    [0.5, 0.5, 0.5],   # 46アイシン社員
    [0.5, 0.5, 0.5],   # 47アイシン社員
    [0.8, 0.8, 0.7],   # 48政治家
    [1.0, 1.0, 1.0],   # 49プロ野球選手
    [1.0, 1.0, 1.0],   # 50プロ野球選手
    [1.0, 1.0, 1.0],   # 51プロ野球選手
    [1.0, 1.0, 1.0],   # 52プロサッカー選手
    [1.0, 1.0, 1.0],   # 53プロサッカー選手
    [1.0, 1.0, 1.0],   # 54プロサッカー選手
    [1.0, 1.0, 1.0],   # 55ユーチューバー
    [1.0, 1.0, 1.0]    # 56ユーチューバー

])


logfile='./log/log.csv'
targetpath='./target_image/'
targetexp='png'
#h5File='./bin/my_model-epoch20.h5'
#h5File='./bin/my_model-n19-epoch25.h5'
#h5File='./bin/my_model-n44-epoch17.h5'
h5File='./bin/my_model-n56-epoch17.h5'
facedetect='./bin/haarcascade_frontalface_alt.xml'


def detect_face(image,imageOrg,detect):
#    print(image.shape)
    #opencvを使って顔抽出
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(facedetect)
    # 顔認識の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
    #顔が１つ以上検出された時
    detecf_exec=False
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
#            print(rect)
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if img.shape[0]<64:
#                print("too small")
                continue
            img = cv2.resize(img,(64,64))
            img=np.expand_dims(img,axis=0)
            name = ''
            if detect==True:
                name = detect_who(img,imageOrg,int(x),int(y),int(width),int(height))
                detecf_exec=True
            cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
    #顔が検出されなかった時
#    else:
#        print("no face")

# 検出実行時に顔検出されなかった場合、画像全体を使用する。
    if detecf_exec==False:
        if detect==True:
            height, width, channels = image.shape[:3]
            rect=[0,0,width,height]
#            print(rect)
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            img = cv2.resize(img,(64,64))
            img=np.expand_dims(img,axis=0)
            name = detect_who(img,imageOrg,rect[0],rect[1],rect[2],rect[3])
    return image



def detect_who(img,image,x,y,w,h):
    print([x,y,w,h])
    logdate = datetime.now()
    strDate = logdate.strftime("%Y%m%d_%H%M%S")
    ImageFile=targetpath+strDate+'.'+targetexp
    print('#### image output = .'+ImageFile)
    cv2.imwrite(ImageFile,image)
    #予測
    name=""
    nameNumLabel=model.predict(img)[0]

    matRecog = nameNumLabel
    dream = np.dot(matRecog, matTable).tolist()[0]
#    print(dream.tolist())


    result={}
    result['date'] = strDate
    result['list'] = nameNumLabel.tolist()
    list_dict=[]
    n=0
    for i in result['list']:
        list_dict.append({'no':labels[n],'accuracy':i})
        n+=1
    result['rank']=sorted(list_dict,key=lambda x:x['accuracy'],reverse=True)
    result['top']=result['rank'][0]['no']
    result['dream']=dream
    result['rect']=[x,y,w,h]
    print(result)
    with open(logfile,'a') as f:
        writer = csv.writer(f)
        f.write(logdate.strftime("%Y/%m/%d %H:%M:%S, "))
        f.write(ImageFile+', ')
        f.write(str(x)+', '+str(y)+', '+str(w)+', '+str(h)+', ')
        writer.writerow(nameNumLabel.tolist()+dream)
    if tcpsend==True:
#        print('Starting the client at', datetime.now())
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(address)
        client.sendall(json.dumps(result).encode('utf-8'))
        data = client.recv(max_size)
#        print('At', datetime.now(), 'someone replied', data)
        client.close()
    if websocketsend==True:
#        print("create websock")
        ws = create_connection(websocketaddress)
#        print("Sending 'Hello, World'...")
        ws.send(json.dumps({"type":"recog","data":result}).encode('utf-8'))
#        rc =  ws.recv()
#        print("Received '%s'" % rc)
        ws.close()
    return result['top']


argv = sys.argv
argc = len(argv)
if (argc > 3):
    #引数がちゃんとあるかチェック
    #正しくなければメッセージを出力して終了
    print('Usage: python3 %s [h5File] [imageFile]' % argv[0])
    print('Example: python3  %s ./bin/my_model-n19-epoch25.h5 ./target_image/20180929_064525.png' % argv[0])
    quit()

if (argc >= 2):
    h5File=argv[1]
model = load_model(h5File)

JPGFile=''
if (argc >= 3):
    JPGFile=argv[2]

# 画像指定があれば、その画像で検出して終了
if JPGFile!='':
	image=cv2.imread(JPGFile)
	image_for_result = image.copy()

	if image is None:
		print("Not open:")
		quit()
	b,g,r = cv2.split(image)
	image = cv2.merge([r,g,b])
	whoImage=detect_face(image,image_for_result,True)
	quit()

# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter
print("[INFO] starting the video stream and FPS counter...")
vs = VideoStream(0).start()
time.sleep(1)
fps = FPS()
fps.start()

# loop over frames from the video file stream
det=False
while True:
	try:
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		det=False
		if key == ord("c"):
			det=True

		# grab the frame from the threaded video stream
		# make a copy of the frame and resize it for display/video purposes
		frame = vs.read()
		image_for_result = frame.copy()
		# flip
		image_for_result = cv2.flip(image_for_result, 1)

		# detect
		b,g,r = cv2.split(image_for_result)
		image = cv2.merge([r,g,b])
		whoImage=detect_face(image,image_for_result,det)

		r,g,b = cv2.split(whoImage)
		whoImage = cv2.merge([b,g,r])

		cv2.imshow("Output", whoImage)

		# update the FPS counter
		fps.update()
	
	# if "ctrl+c" is pressed in the terminal, break from the loop
	except KeyboardInterrupt:
		break

	# if there's a problem reading a frame, break gracefully
	except AttributeError:
		break

# stop the FPS counter timer
fps.stop()

# destroy all windows if we are displaying them
cv2.destroyAllWindows()

# stop the video stream
vs.stop()

# display FPS information
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


