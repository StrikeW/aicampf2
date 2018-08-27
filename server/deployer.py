import os
from server import config
from server.dal.entities import Model
from server.utils.process import exec_cmd


# deploy a model as a HTTP service
def deploy_model(mid):
    # TODO
    # model = Model.get_by_id(mid)

    pid = os.fork()
    if pid == 0:    # child
        exec_cmd(cmd=['./server/image_service.py', '127.0.0.1', '9090'],
                 stream_output=True)
    print('deploy ok')
    return True
