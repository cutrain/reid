import cv2
import zmq
import time
import json
import binascii
import threading
import multiprocessing

DEBUG = True
processing_manager = None

function_port = '5565'
control_port = '5566'
pulse_port = '5567'
result_port = '5568'


if DEBUG:
    class auto_mark_iter:
        def __init__(self, path):
            self.path = path
        def __len__(self):
            return 100
        def __next__(self):
            for i in range(100):
                time.sleep(0.5)
                yield {
                    'frame_index':i,
                    'frame_result':{
                        'object_num':1,
                        'coordinate_matrix':[[20,70,10,30]],
                        'id':1,
                    }
                }
        def __iter__(self):
            yield from self.__next__()
else:
    from reid import auto_mark_iter

def checksum(ret):
    return binascii.crc32(json.dumps(ret, sort_keys=True).encode('utf-8'))

def func_control(req):
    try:
        print('1')
        chsum = req.pop('chsum')
        if chsum != checksum(req):
            return False
        head = req.pop('head')
        print('2')
        alg_id = req.pop('alg_id')
        print('3', head)
        if head != 'control':
            return False
        print('4', alg_id)
        if alg_id != 2001:
            return False
        global processing_manager
        print('5', processing_manager)
        if processing_manager is None:
            return False
        processing_manager.terminate()
        print('Function Stoped')
        processing_manager = None
        return True
    except Exception:
        return False

def data_load(filepath, eta):
    # NOTE : this function not used
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.bind('tcp://' + platform_ip + ":" +platform_port)
    # TODO : chsum
    etatime = str(eta//3600) + ":" + str((eta // 60) % 60) + ":" + str(eta % 60)
    result = {
        'head':'report',
        'file':filepath,
        'time':etatime,
        'alg_id':2000,
    }
    result['chsum'] = checksum(result)
    socket.send_json(result)
    ret = socket.recv_json()
    func_control(ret)

def pulse(pack):
    global pulse_port
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind('tcp://*:'+pulse_port)
    start = time.time()
    while True:
        time.sleep(5)
        end = time.time()
        passed = end-start
        remain = passed / (pack['progress']+1e-5) * pack['total']
        passed = int(passed)
        remain = int(remain)
        passed = "{:0>2d}{:0>2d}{:0>2d}".format(passed//3600, (passed//60)%60, passed%60)
        remain = "{:0>2d}{:0>2d}{:0>2d}".format(remain//3600, (remain//60)%60, remain%60)
        ret = {
            'head':'msg',
            'file':pack['file'],
            'time_pass':passed,
            'time_reamin':remain,
            'time':time.strftime("%H%M%S", time.localtime()),
            'alg_id':2000,
        }
        ret['chsum'] = checksum(ret)
        print('Pulse send : {}'.format(ret))
        socket.send_json(ret)

def solve_reid(path):
    iter = auto_mark_iter(path)
    pulse_pack = {
        'progress':0,
        'total':len(iter),
        'file':path,
    }
    pulse_thread = threading.Thread(target=pulse, args=(pulse_pack,))
    pulse_thread.daemon = True
    pulse_thread.start()
    # TODO: pulse get result
    global result_port
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind('tcp://*:'+ result_port)
    for i in iter:
        ret = {
            'head':'data',
            'alg_id':2000,
            'file':path,
            'frame_index':i['frame_index'],
            'frame_result':i['frame_result'],
        }
        pulse_pack['progress'] += 1
        ret['chsum'] = checksum(ret)
        print('Result send : {}'.format(ret))
        socket.send_json(ret)



def func_cmd(req):
    global processing_manager
    ret = {
        'head':'rec',
        'file':'',
        'network':0,
        'ready':0,
        'alg_id':2000,
        'error':400,
    }
    try:
        print('1')
        chsum = req.pop('chsum')
        verify = checksum(req)
        if verify != chsum:
            print('get checksum : {}, but calculated : {}'.format(chsum, verify))
            raise KeyError
        print('2')
        alg_id = req.pop('alg_id')
        if alg_id != 2001:
            raise TypeError
        print('3')
        path = req.pop('file')
        print('4', path)
        cap_test = cv2.VideoCapture(path)
        print('5')
        if cap_test.isOpened() == False:
            print('6')
            raise FileNotFoundError
        else:
            print('7')
            cap_test.release()
        if processing_manager is not None:
            print('8')
            processing_manager.terminate()
            processing_manager = None
        print('9')
        processing_manager = multiprocessing.Process(target=solve_reid, args=(path,))
        processing_manager.daemon = True
        processing_manager.start()
        ret.update({
            'error':200,
            'ready':1,
        })
    except TypeError as e:
        ret.update({'error':400})
    except KeyError as e:
        ret.update({'error':400})
    except FileNotFoundError as e:
        ret.update({'error':404})
    except MemoryError as e:
        ret.update({'error':502})
    except Exception as e:
        ret.update({'error':500})
        print('ERROR', e)
    return ret

def func_server():
    global function_port
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:'+function_port)
    print('Function Server start')
    while True:
        req = socket.recv_json()
        print('Function Server Receive : {}'.format(req))
        ret = func_cmd(req)
        ret['chsum'] = checksum(ret)
        print('Function Server Send : {}'.format(ret))
        socket.send_json(ret)

def control_server():
    global control_port
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:'+control_port)
    print('Control Server start')
    while True:
        req = socket.recv_json()
        print('Control Server Receive : {}'.format(req))
        stoped = func_control(req)
        ret = {
            'head':'stop',
            'alg_id':2000,
            'succeed':1 if stoped else 0,
        }
        ret['chsum'] = checksum(ret)
        print('Control Server Send : {}'.format(ret))
        socket.send_json(ret)


if __name__ == '__main__':
    try:
        server_list = [func_server, control_server]
        arg_list = [(), ()]
        thread_list = []
        for i,j in zip(server_list, arg_list):
            server = threading.Thread(target=i, args=j)
            server.daemon = True
            thread_list.append(server)
        for i in thread_list:
            i.start()
        for i in thread_list:
            i.join()
    except KeyboardInterrupt:
        for i in thread_list:
            i.terminate()
