from dotenv import load_dotenv
import os
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
import re
import base64
import time
from mimetypes import guess_extension, guess_type

# My files
import train
import test

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
PORT = os.getenv("PORT")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/train", methods=['POST'])
def train_endpoint():
    genFiles = request.files.getlist("genFiles")
    fakeFiles = request.files.getlist("fakeFiles")

    name = request.form['model']

    if not os.path.exists(name):
        os.mkdir(name)

    if not os.path.exists(name+'/gen'):
        os.mkdir(name+'/gen')

    if not os.path.exists(name+'/fake'):
        os.mkdir(name+'/fake')

    for i, file in enumerate(genFiles):

        file.save(os.path.join(name+'/gen', 'gen.' + str(i+1) + '.' + file.mimetype.split('/')[1]))

    for i, file in enumerate(fakeFiles):
        file.save(os.path.join(name+'/fake', 'fake.' + str(i+1) + '.' + file.mimetype.split('/')[1]))
    
    
    train.train_model(name)

    return jsonify ({
        'msg': "Training finish"
    })

@app.route("/get", methods=['GET'])
def get():
    models = []

    for dir in os.listdir('./'):
        if not os.path.isfile(dir) and not dir.startswith('_'):
            print(dir, flush=True)
            models.append(dir)


    return jsonify ({
        'models': models
    })

@app.route("/test", methods=['POST'])
def test_endpoint():
    testFiles = request.files.getlist("testFiles")

    name = request.form['model']

    if not os.path.exists(name):
        os.mkdir(name)

    if not os.path.exists(name+'/test'):
        os.mkdir(name+'/test')

    for i, file in enumerate(testFiles):
        file.save(os.path.join(name+'/test', 'test.' + str(i+1) + '.' + file.mimetype.split('/')[1]))
    
    SVM_result, MLP_result, MV5_result = test.test_model(name)

    return jsonify ({
        'svm': SVM_result,
        'mlp' : MLP_result,
        'mv5': MV5_result
    })

@app.route("/add", methods=['POST'])
def add_endpoint():
    name = request.form['model']

    if not os.path.exists(name):
        os.mkdir(name)

    return jsonify ({
        'msg': 'Model created'
    })

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=PORT) # for running on localhost
    # app.run(ssl_context=('cert.pem', 'key.pem'), host='0.0.0.0', port=PORT) # for deployment w/ ssl cert
