# image service client
from server.service import ImageService
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

class ImageCli(object):
    def __init__(self):
        """TODO: to be defined1. """
        self.state = 'close'

    def connect(self):
        if self.state == 'connected':
            return

        # Make socket
        self.transport = TSocket.TSocket('localhost', 9090)
        # Buffering is critical. Raw sockets are very slow
        self.transport = TTransport.TBufferedTransport(self.transport)
        # Wrap in a protocol
        protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        # Create a client to use the protocol encoder
        self.client = ImageService.Client(protocol)
        # Connect!
        self.transport.open()
        print('ImageClient: Connected')
        return True

    def close(self):
        if self.state == 'close':
            return

        self.transport.close()
        print('ImageClient: Close')


if __name__ == "__main__":
    imgCli = ImageCli()
    imgCli.connect()
    going = True
    while going:
        cmd = input('Enter a command:')
        if cmd == 'ping':
            resp = imgCli.client.ping()
            print('server resp: %s' % resp)
        elif cmd == 'predict':
            predicts = imgCli.client.predict([1, 1, 0])
            print('server resp: ', predicts)
        elif cmd == 'sum':
            ret = imgCli.client.sum([5,10,2])
            print('server resp: sum=%d' % ret)
        elif cmd == 'quit':
            going = False
    imgCli.close()





