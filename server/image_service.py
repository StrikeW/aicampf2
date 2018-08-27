#!/usr/bin/env python
# -*- coding: utf-8 -*-

# image service
from service import ImageService
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

class ImageHandler:
    def __init__(self):
        self.log = {}

    def ping(self):
        return 'ImageService: I am alive'

    def predict(self, datas):
        print('predict: %d examples' % len(datas))
        return [1,2,3]

def run_service(_host, _port):
    handler = ImageHandler()
    processor = ImageService.Processor(handler)
    transport = TSocket.TServerSocket(host=_host, port=int(_port))
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    server.serve()

if __name__ == "__main__":
    print('Thrift Server start at host=127.0.0.1, port=9090')
    run_service('127.0.0.1', 9090)

