Web�J�����ŎB�e��(c�L�[)�APNG��target_image�֕ۑ��A���_+�h���[���̃}�g���b�N�X���Z�����{��CSV(log.csv),JSON(�W���o��+WebSocket)�֏o�́�WebSocketServer��HTML�փf�[�^���M���O���t��\������PNG�摜��WebSocketServer�֑��M���O���t��ۑ�(PNG)
2018/10/03
�E�O���t�ƃ��[�_�`���[�g�𓝍�

�g����(�ȉ��̃R�}���h)
A.�V�X�e�������グ�̏ꍇ
> ./00run.sh 
WebSocketServer�𗧂��グ�AHTML��\�����AWeb�J�������g�p���Ċ�F�������܂��B

B.�ʂɊ�F�������グ�̏ꍇ
Web�J�������g�p����ꍇ�AWindows��ŃJ������������AVirtualBox��Web�J������L��������B
> python3 ./bin/06test-o.py
�摜�t�@�C�����g�p����ꍇ
> python3 ./bin/06test-o.py ./bin/my_model-n56-epoch17.h5 ./target_image/20180929_064525.png 

CSV���O�ۑ���
./log

�摜�ۑ���
./target_image

��
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
Web�J����

Web�u���E�U(firefox) 62.0 (64 �r�b�g)



#sudo pip3 install git+https://github.com/Pithikos/python-websocket-server
#sudo pip3 install websocket-server
#sudo pip3 install websocket-client

dscope-system2.png�̐Ԙg+�Θg�܂Ŏ��{

06test-o.py
����
tcpsend=False
��
tcpsend=True
�ɂ��邱�ƂŁASocket�Ńf�[�^�𑗕t���܂��B


