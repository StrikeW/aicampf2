#!/usr/bin/env python
# -*- coding: utf-8 -*-
from server import _run_server, app

if __name__ == "__main__":
    # host = sys.argv[1]
    # port = sys.argv[2]
    #_run_server('127.0.0.1', 4000, 1, None)
    app.run(port=4000, debug=True)
