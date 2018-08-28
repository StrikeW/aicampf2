import json
import time
import cv2
import os
import numpy as np
from server.model import cnn_mnist
from server.model import cnn_cifar
from flask import Response, request
from server.image_cli import ImageCli
from server import config

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


def _hello():
    resp = Response(mimetype='application/json')
    resp.set_data(u'{"hello": "world"}');
    return resp

def _train_model(name, conf, data):
    if name == "CNN":
        dic = json.loads(conf)
        cnn_mnist.set_parameter(dic)
        train_ret = cnn_cifar.train(data)
        m = Model()
        m.saved_path = train_ret['save_path']
        m.type = 1
        m.name = 'cnn'
        m.hyper_params = conf
        m.accuracy = train_ret['acc']

        with DBSession() as sess:
            sess.add(m)
            sess.commit()
            print('Insert a trained model. id=%d' % m.mid)

        train_ret['mid'] = m.mid

        ans = {}
        ans['Conf'] = dic
        ans['Result'] = train_ret

        return json.dumps(ans, cls=MyEncoder)

def _train(req=request):
    if req.method == "POST":
        print(len(req.files))
        data = req.files['datafile']
        conf = req.form['conf']
        name = req.form['model']
        return _train_model(name, conf, data)

    resp = Response(mimetype='text/plain')
    resp.set_data(u'Task submit successful!');
    return resp

def _deploy(req = request):
    if req.method == "POST":
        mid = req.form['mid']
        ret = deployer.deploy_model(mid)

    resp = Response(mimetype='application/json')
    resp.set_data(u'{"code": 0, "msg": "depoly successful"}');
    return resp



serv_clis = []

def _img_ping():
    global serv_clis
    if len(serv_clis) == 0:
        imgcli = ImageCli()
        imgcli.connect()
        serv_clis.append(imgcli)
    else:
        imgcli = serv_clis[0]

    ret = imgcli.client.ping()
    resp = Response(mimetype='application/json')
    resp.set_data(u'{"state": "%s"}' % ret);
    return resp


def _img_predict(req=request):
    f = req.files['testfile']
    file_path = os.path.join(config.upload_path, f.filename)
    f.save(os.path.join(config.upload_path, f.filename))

    print('img_predict: file saved: %s' % file_path)

    global serv_clis
    if len(serv_clis) == 0:
        imgcli = ImageCli()
        imgcli.connect()
        serv_clis.append(imgcli)
    else:
        imgcli = serv_clis[0]

    ret = imgcli.client.predict(file_path)
    print('_img_predict: prediction=%s' % ret)

    resp = Response(mimetype='application/json')
    resp.set_data(u'{"code": 0, "result": "%s"}' % ret[0]);
    return resp

def get_endpoints():
    """
    :return: List of tuples (path, handler, methods)
    """
    # a fake handler
    ret = [('/api/hello', HANDLERS['hello'], ['GET']),
            ('/api/train', HANDLERS['train'], ['POST']),
            ('/api/deploy', HANDLERS['deploy'], ['POST', 'GET']),
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
