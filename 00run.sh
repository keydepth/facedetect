#!/bin/bash

# tensolflowを立ち上げる
#cd ~/tensorflow/
#source ./tensorflow/bin/activate

# 新しいターミナルを開いてWebSocketサーバを立ち上げる。
# 役割:
# ①検出結果を受取り、ブラウザへ転送する。
# ②ブラウザで表示されたチャートを画像保存する。
gnome-terminal -e 'python3 ./bin/20webSocketServer.py' &


# 役割:
# ブラウザでレーダーチャートを開いて、WebSocketサーバへ接続する
#firefox -new-window ./html/radar-dscope.html &
#firefox -new-window ./html/bar-dscope.html &
firefox -new-window ./html/mix-dscope.html &

# Tensolflow環境で、顔検出、画像保存、JSON出力、WebSocketサーバへ検出結果の送付
#gnome-terminal -e 'python3 ./bin/06test-o.py' &
python3 ./bin/06test-o.py
