# Define all the service endpoint handlers here.
import json
import re
import six
import sys

import numpy as np
from server.model import cnn_mnist
from flask import Response, request
from google.protobuf.json_format import ParseDict
from querystring_parser import parser
from server.image_cli import ImageCli

import server.deployer as deployer

from server.dal.entities import Model, Deploy
from server.dal import DBSession


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj), 4)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def _not_implemented():
    response = Response()
    response.status_code = 404
    return response

def _hello():
    resp = Response(mimetype='application/json')
    resp.set_data(u'{"hello": "world"}');
    return resp

def _train_model(name, conf):
    if name == "CNN":
        dic = json.loads(conf)
        cnn_mnist.set_parameter(dic)
        train_ret = cnn_mnist.train()

        m = Model()
        m.saved_path = train_ret['save_path']
        m.type = 1
        m.name = 'cnn'
        m.hyper_params = conf
        m.accuracy = train_ret['acc']

        with DBSession() as sess:
            sess.add(m)
            sess.commit()
            sess.close()
            print('Insert a trained model. id=%d' % m.mid)

        train_ret['mid'] = m.mid

        ans = {}
        ans['Conf'] = dic
        ans['Result'] = train_ret

        return json.dumps(ans, cls=MyEncoder)

def _train(req=request):
    if req.method == "POST":
        print(len(req.files))
        # data = req.files['datafile']
        conf = req.form['conf']
        name = req.form['model']
        return _train_model(name, conf)

    resp = Response(mimetype='text/plain')
    resp.set_data(u'Task submit successful!');
    return resp

def _deploy(req = request):
    mid = req.form['mid']

    ret = deployer.deploy_model(mid = 5)
    resp = Response(mimetype='application/json')
    resp.set_data(u'{"code": 0, "msg": "depoly successful"}');
    return resp

def _img_predict(req=request):
    f = req.files['testfile'].read()
    img = list(f)
    # TODO
    imgcli = ImageCli()
    imgcli.connect()

    ret = imgcli.client.ping()
    print('_img_predict: ret=%s' % ret)

    resp = Response(mimetype='application/json')
    resp.set_data(u'{"code": 0, "msg": "%s"}' % ret);
    return resp

def get_endpoints():
    """
    :return: List of tuples (path, handler, methods)
    """
    # a fake handler
    ret = [('/api/hello', HANDLERS['hello'], ['GET']),
            ('/api/train', HANDLERS['train'], ['POST']),
            ('/api/deploy', HANDLERS['deploy'], ['POST']),
            ('/api/img_predict', HANDLERS['img_predict'], ['POST', 'GET']),
            ]
    return ret


# dict of handlers
HANDLERS = {
        "hello": _hello,
        'train': _train,
        'deploy': _deploy,
        'img_predict': _img_predict,
}
