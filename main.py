import joblib
from flask import Flask, request
from markupsafe import escape
import sklearn
import pandas as pd

app = Flask(__name__)


@app.route("/<name>")
def hello(name):
    return f"Hello, {escape(name)}!"


@app.route('/recognize', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        text = request.form['text']
    else:
        text = request.args.get('text')
    X = pd.DataFrame([[text]])
    X.columns = ['text']
    X = joblib.load("./cvec.joblib").transform(X)
    X = joblib.load("./tfid.joblib").transform(X)
    tr = joblib.load("./rf.joblib").predict(X)
    return {0: 'Средняя значимость', 1: 'Низкая значимость', 2: 'Очень важно'}[tr[0]]


def main():
    return app
