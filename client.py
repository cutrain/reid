import zmq
import time
import json
import binascii
import threading

# server = '180.167.245.194'
# testfile = './a.avi'
# server = '101.89.135.165'
# testfile = './a.avi'
server = 'localhost'
testfile = './a.avi'

def checksum(js):
    return binascii.crc32(json.dumps(js, sort_keys=True).encode('utf-8'))

def start():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://' + server + ':5565')
    req = {
        'head':'cmd',
        'file':testfile,
        'alg_id':2001,
    }
    chsum = checksum(req)
    req.update({'chsum':chsum})
    print('start send {}'.format(req))
    socket.send_json(req)
    message = socket.recv_json()
    print('start recv {}'.format(message))

def stop():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://' + server + ':5566')
    req = {
        'head':'control',
        'alg_id':2001,
    }
    chsum = checksum(req)
    req.update({'chsum':chsum})
    print('stop send {}'.format(req))
    socket.send_json(req)
    message = socket.recv_json()
    print('stop recv {}'.format(message))

def result_client():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://' + server + ':5568')
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    while True:
        recv = socket.recv_json()
        print('Result Client recv :{}'.format(recv))

def pulse_client():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://' + server + ':5567')
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    while True:
        recv = socket.recv_json()
        print('Pulse Client recv :{}'.format(recv))


if __name__ == "__main__":
    try:
        client_list = [result_client, pulse_client]
        thread_list = []
        for i in client_list:
            thread = threading.Thread(target=i)
            thread.daemon = True
            thread_list.append(thread)
        for i in thread_list:
            i.start()
        start()
        time.sleep(1000)
        stop()
        for i in thread_list:
            i.join()
    except Exception:
        exit()
