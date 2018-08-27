import os
import time
from server import config
from server.dal.entities import Model
from server.utils.process import exec_cmd


# deploy a model as a HTTP service
def deploy_model(mid):
    # TODO
    # model = Model.get_by_id(mid)

    try:
        pid = os.fork()
    except Exception as e:
        raise e

    if pid == 0:    # child
        print('I am child')
        exec_cmd(cmd=['./server/image_service.py', mid], stream_output=True)

    time.sleep(2)
    print('deploy ok')
    return True
