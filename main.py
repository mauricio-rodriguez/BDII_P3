from flask import Flask,render_template, request, Response, redirect,url_for
import json
from search import knn_rtree, knn_sequential
import os

app = Flask(__name__)

allRows = 13170


@app.route("/search", methods = ['POST'])
def search():
    if 'file' not in request.files:
        print("no file part")
    else: 
        file = request.files['file']
        if file.filename == '':
            return "No hay archivo"
        if file:
            file_dir = os.path.join("input_images/", file.filename)
            file.save(file_dir)
            closest_neighbors = knn_rtree(13170, file_dir)
            print(closest_neighbors)
        return render_template('index.html', filename = file.filename)

        

@app.route("/")
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.secret_key = ".."
    app.run(port=3000, threaded=True, host=('127.0.0.1'))