# Define all the service endpoint handlers here.
import json
import os
import re
import six

from flask import Response, request, send_file
from google.protobuf.json_format import MessageToJson, ParseDict
from querystring_parser import parser

def _get_request_message(request_message, flask_request=request):
    if flask_request.method == 'GET' and len(flask_request.query_string) > 0:
        # This is a hack to make arrays of length 1 work with the parser.
        # for example experiment_ids%5B%5D=0 should be parsed to {experiment_ids: [0]}
        # but it gets parsed to {experiment_ids: 0}
        # but it doesn't. However, experiment_ids%5B0%5D=0 will get parsed to the right
        # result.
        query_string = re.sub('%5B%5D', '%5B0%5D', flask_request.query_string.decode("utf-8"))
        request_dict = parser.parse(query_string, normalized=True)
        ParseDict(request_dict, request_message)
        return request_message

    request_json = flask_request.get_json(force=True, silent=True)

    # Older clients may post their JSON double-encoded as strings, so the get_json
    # above actually converts it to a string. Therefore, we check this condition
    # (which we can tell for sure because any proper request should be a dictionary),
    # and decode it a second time.
    if isinstance(request_json, six.string_types):
        request_json = json.loads(request_json)

    # If request doesn't have json body then assume it's empty.
    if request_json is None:
        request_json = {}
    ParseDict(request_json, request_message)
    return request_message


def _not_implemented():
    response = Response()
    response.status_code = 404
    return response


# def _message_to_json(message):
    # # preserving_proto_field_name keeps the JSON-serialized form snake_case
    # return MessageToJson(message, preserving_proto_field_name=True)

_TEXT_EXTENSIONS = ['txt', 'yaml', 'json', 'js', 'py', 'csv', 'md', 'rst', 'MLmodel', 'MLproject']


# def _get_run():
    # request_message = _get_request_message(GetRun())
    # response_message = GetRun.Response()
    # response_message.run.MergeFrom(_get_store().get_run(request_message.run_uuid).to_proto())
    # response = Response(mimetype='application/json')
    # response.set_data(_message_to_json(response_message))
    # return response


# def _get_paths(base_path):
    # """
    # A service endpoints base path is typically something like /preview/mlflow/experiment.
    # We should register paths like /api/2.0/preview/mlflow/experiment and
    # /ajax-api/2.0/preview/mlflow/experiment in the Flask router.
    # """
    # return ['/api/2.0{}'.format(base_path), '/ajax-api/2.0{}'.format(base_path)]

def _hello():
    resp = Response(mimetype='application/json')
    resp.set_data(u'{"hello": "world"}');
    return resp

def _train():
    resp = Response(mimetype='text/plain')
    resp.set_data(u'Task submit successful!');
    return resp

def get_endpoints():
    """
    :return: List of tuples (path, handler, methods)
    """
    # a fake handler
    ret = [('/api/hello', HANDLERS['hello'], ['GET']),
            ('/api/train', HANDLERS['train'], ['POST'])]
    return ret


# dict of handlers
HANDLERS = {
        "hello": _hello,
        'train': _train
    # CreateExperiment: _create_experiment,
    # GetExperiment: _get_experiment,
    # CreateRun: _create_run,
    # UpdateRun: _update_run,
    # LogParam: _log_param,
    # LogMetric: _log_metric,
    # GetRun: _get_run,
    # SearchRuns: _search_runs,
    # ListArtifacts: _list_artifacts,
    # GetArtifact: _get_artifact,
    # GetMetricHistory: _get_metric_history,
    # ListExperiments: _list_experiments,
    # GetParam: _get_param,
    # GetMetric: _get_metric,
}
