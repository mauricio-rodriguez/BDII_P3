from flask import Flask,render_template, request, Response, redirect,url_for
import json
from search import knn_rtree
import os

app = Flask(__name__)


@app.route("/search", methods = ['POST'])
def search():
        if 'file' not in request.files:
            print("no file part")
        else: 
            file = request.files['file']
            if file.filename == '':
                return "No hay archivo"
            if file:
                file.save(os.path.join("input_images/", file.filename))
                
            return render_template('index.html', filename = file.filename)

        

@app.route("/")
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.secret_key = ".."
    app.run(port=3000, threaded=True, host=('127.0.0.1'))