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

def detect_face(image):
    print(image.shape)
    #opencvを使って顔抽出
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    # 顔認識の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
    #顔が１つ以上検出された時
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue
            img = cv2.resize(image,(64,64))
            img=np.expand_dims(img,axis=0)
            name = detect_who(img)
            cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
    #顔が検出されなかった時
    else:
        print("no face")
    return image
    
def detect_who(img):
    #予測
    name=""
    print(model.predict(img))
    nameNumLabel=np.argmax(model.predict(img))
    name=str(nameNumLabel+1)
    return name

model = load_model('./my_model-n19-epoch100.h5')


# image=cv2.imread("./origin_image/00.3.jpg")
image=cv2.imread("./test.jpg")
#image=cv2.imread("./origin_image/00.IMG_5604.jpg")
if image is None:
    print("Not open:")
b,g,r = cv2.split(image)
image = cv2.merge([r,g,b])
whoImage=detect_face(image)

plt.imshow(whoImage)
plt.show()

#quit()

# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter
print("[INFO] starting the video stream and FPS counter...")
vs = VideoStream(0).start()
time.sleep(1)
fps = FPS()
fps.start()

# loop over frames from the video file stream
while True:
	try:
		# grab the frame from the threaded video stream
		# make a copy of the frame and resize it for display/video purposes
		frame = vs.read()
		image_for_result = frame.copy()

		# detect
		b,g,r = cv2.split(image_for_result)
		image = cv2.merge([r,g,b])
		whoImage=detect_face(image)

		r,g,b = cv2.split(whoImage)
		whoImage = cv2.merge([b,g,r])

		cv2.imshow("Output", whoImage)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

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


