import os

from flask import Flask, send_from_directory, make_response, render_template

from server import handlers
from utils.process import exec_cmd

FILE_STORE_ENV_VAR = "MLFLOW_SERVER_FILE_STORE"
ARTIFACT_ROOT_ENV_VAR = "MLFLOW_SERVER_ARTIFACT_ROOT"
STATIC_PREFIX_ENV_VAR = "MLFLOW_STATIC_PREFIX"

REL_STATIC_DIR = "templates"
app = Flask(__name__)
# app.root_path is 'server/'
STATIC_DIR = os.path.join(app.root_path, REL_STATIC_DIR)

for http_path, handler, methods in handlers.get_endpoints():
    app.add_url_rule(http_path, handler.__name__, handler, methods=methods)

def _add_static_prefix(route):
    prefix = os.environ.get(STATIC_PREFIX_ENV_VAR)
    if prefix:
        return prefix + route
    return route

# Serve the font awesome fonts for the React app
@app.route(_add_static_prefix('/webfonts/<path:path>'))
def serve_webfonts(path):
    return send_from_directory(STATIC_DIR, os.path.join('webfonts', path))


# We expect the react app to be built assuming it is hosted at /static-files, so that requests for
# CSS/JS resources will be made to e.g. /static-files/main.css and we can handle them here.
@app.route(_add_static_prefix('/static-files/<path:path>'))
def serve_static_file(path):
    return send_from_directory(STATIC_DIR, path)

def load_model():
    print('Root path %s' % app.root_path)
    a = []
    fpath = os.path.join(STATIC_DIR, 'model_list.txt')
    with open(fpath) as f:
        for line in f:
            print(line)
            line = line.strip()
            a.append(line)
    return a

#def load_config(conf_path):
#    fpath = os.path.join(STATIC_DIR, conf_path)
#    with open(fpath) as f:
#        a = f.read()
#        return a

# Serve the index.html for the React App for all other routes.
@app.route('/models')
def show_models():
    models = load_model()
#    model_config = []
#    for n in models:
#        model_config.append(load_config(n))

    return render_template('tmpl.html', my_string="Wheeeee!", my_list=[1,2,3,4], my_models=models)

@app.route('/test', methods=["POST", "GET"])
def show_test():
    return render_template('test.html')

@app.route('/')
def serve():
    return send_from_directory(STATIC_DIR, 'index.html')


# def _run_server(file_store_path, default_artifact_root, host, port, workers, static_prefix):
def _run_server(host, port, workers, static_prefix):
    """
    Run the MLflow server, wrapping it in gunicorn
    :param static_prefix: If set, the index.html asset will be served from the path static_prefix.
                          If left None, the index.html asset will be served from the root path.
    :return: None
    """
    env_map = {}
    # if file_store_path:
        # env_map[FILE_STORE_ENV_VAR] = file_store_path
    # if default_artifact_root:
        # env_map[ARTIFACT_ROOT_ENV_VAR] = default_artifact_root
    if static_prefix:
        env_map[STATIC_PREFIX_ENV_VAR] = static_prefix
    bind_address = "%s:%s" % (host, port)
    exec_cmd(["gunicorn", "-b", bind_address, "-w", "%s" % workers, "server:app"],
             env=env_map, stream_output=True)
