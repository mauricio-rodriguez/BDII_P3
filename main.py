from flask import Flask,render_template, request, Response, redirect,url_for
import json
from search import knn_rtree, knn_sequential
# from generate_tree import generate_tree
import os
from rtree import index

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
            prop = index.Property()
            prop.dimension = 58
            prop.buffering_capacity = 5
            tree_dir = "./index/rtree_"+str(13170)
            idx = index.Index(tree_dir, properties = prop)
            closest_neighbors = knn_rtree(idx, 13170, file_dir)
        return render_template('index.html', filename = file.filename)

        

@app.route("/")
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.secret_key = ".."
    app.run(port=3000, threaded=True, host=('127.0.0.1'))