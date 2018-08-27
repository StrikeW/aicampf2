#!/usr/bin/env python
# -*- coding: utf-8 -*-

# image service
import sys
import cv2
import numpy as np
from service import ImageService
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from dal.entities import Model, Deploy
from dal import DBSession
from tensorflow.contrib import predictor

class ImageHandler:
    def __init__(self, predict_fn):
        self.log = {}
        self.predict_fn = predict_fn
        print('init predictfn ok')

    def ping(self):
        return 'MNISTService: I am alive'

    def predict(self, filepath):
        img = cv2.imread(filepath)
        img = cv2.resize(img, (28,28), cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_sz = 28*28
        img = img.reshape(img_sz)
        x_test = np.ndarray((1, img_sz), dtype=np.float32)
        x_test[0] = img

        predictions = self.predict_fn( {"images": x_test} )
        print(predictions)
        return predictions['output']

def load_model(mid):
    predict_fn = None
    with DBSession() as sess:
        for m in sess.query(Model).filter(Model.mid == mid):
            print('get model: saved_path: %s' % m.saved_path)
            predict_fn = predictor.from_saved_model(m.saved_path)
            # handler.set_predictfn(predict_fn)
            # predictions = predict_fn( {"images": x_test} )
            # df = pd.DataFrame(predictions)
    return predict_fn

def run_service(_host, _port, predict_fn):
    handler = ImageHandler(predict_fn)
    processor = ImageService.Processor(handler)
    transport = TSocket.TServerSocket(host=_host, port=int(_port))
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    server.serve()

if __name__ == "__main__":
    print('argc: %d' % len(sys.argv))
    print('argv: %s' % sys.argv)
    print('MNIST service start at host=127.0.0.1, port=9090')
    mid = int(sys.argv[1])
    predict_fn = load_model(mid)
    run_service('127.0.0.1', 9090, predict_fn)

