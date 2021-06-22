#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/19 13:35
# @Author: chenhr33733
# @File: app.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


import json
import flask
from flask import Flask, request
from flask_cors import CORS
from gevent import pywsgi


app = Flask('temp')
CORS(app, supports_credentials=True)
app.secret_key = 'chenhr33733'


@app.route('/temp', methods=['GET', 'POST'])
def temp():
    if request.method == 'POST':
        sentence = flask.request.form['sentence']
        res = {'status': 'SUCCESS', 'body': sentence}
        res = json.dumps(res)
        return res


if __name__ == '__main__':
    app_host = '172.27.28.196'
    app_port = 5555
    # app.run(host=app_host, port=app_port, threaded=True)
    server = pywsgi.WSGIServer((app_host, app_port), app)
    server.serve_forever()