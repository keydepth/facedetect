Webカメラで撮影し(cキー)、PNGをtarget_imageへ保存、推論+ドリームのマトリックス演算を実施→CSV(log.csv),JSON(標準出力+WebSocket)へ出力→WebSocketServer→HTMLへデータ送信→グラフを表示してPNG画像をWebSocketServerへ送信→グラフを保存(PNG)
2018/10/03
・グラフとレーダチャートを統合

使い方(以下のコマンド)
A.システム立ち上げの場合
> ./00run.sh 
WebSocketServerを立ち上げ、HTMLを表示し、Webカメラを使用して顔認識をします。

B.個別に顔認識立ち上げの場合
Webカメラを使用する場合、Windows上でカメラを許可する、VirtualBoxでWebカメラを有効化する。
> python3 ./bin/06test-o.py
画像ファイルを使用する場合
> python3 ./bin/06test-o.py ./bin/my_model-n56-epoch17.h5 ./target_image/20180929_064525.png 

CSVログ保存先
./log

画像保存先
./target_image

環境
VirtualBox 5.2.16 r123759 (Qt5.6.2)
Ubuntu 18.04 desktop
OpenCV 3.4.2
Python 3.6.5
Keras==2.2.2
Keras-Applications==1.0.4
Keras-Preprocessing==1.0.2
opencv-contrib-python==3.4.2.17
opencv-python==3.4.2.17
tensorflow==1.10.1
websocket-client==0.53.0
websocket-server==0.4
numpy==1.14.5
h5py==2.8.0
Webカメラ

Webブラウザ(firefox) 62.0 (64 ビット)



#sudo pip3 install git+https://github.com/Pithikos/python-websocket-server
#sudo pip3 install websocket-server
#sudo pip3 install websocket-client

dscope-system2.pngの赤枠+緑枠まで実施

06test-o.py
内の
tcpsend=False
を
tcpsend=True
にすることで、Socketでデータを送付します。


