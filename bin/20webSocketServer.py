import logging
from websocket_server import WebsocketServer
import base64
import json

#sudo pip3 install git+https://github.com/Pithikos/python-websocket-server
#sudo pip3 install websocket-server
#sudo pip3 install websocket-client

charpath='./target_image/'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(' %(module)s -  %(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
 
# Callback functions
 
def new_client(client, server):
  logger.info('New client {}:{} has joined.'.format(client['address'][0], client['address'][1]))

def client_left(client, server):
  logger.info('Client {}:{} has left.'.format(client['address'][0], client['address'][1]))

def message_received(client, server, message):
#  logger.info('Message "{}" has been received from {}:{}'.format(message, client['address'][0], client['address'][1]))
#  print(message)
  msgObj = json.loads(message)
  if msgObj["type"]=="img":
    outFileName=charpath+msgObj["date"]+"-"+msgObj["subtype"]+".png"
    print('#### chart output = '+outFileName)
#    fh = open("imageToSave.png", "wb")
    fh = open(outFileName, "wb")
    print(message[22:50])
    fh.write(base64.decodebytes(msgObj["data"][22:].encode('utf-8')))
    fh.close()
  if msgObj["type"]=="recog":
    reply_message=json.dumps(msgObj["data"])
    for cl in server.clients:
      if cl['id'] != client['id']:
        server.send_message(cl, message)
      

# Main
if __name__ == "__main__":
  server = WebsocketServer(port=6789, host='localhost', loglevel=logging.INFO)
#  server = WebsocketServer(port=12345, host='localhost')
  server.set_fn_new_client(new_client)
  server.set_fn_client_left(client_left)
  server.set_fn_message_received(message_received)
  server.run_forever()

