"""
	Author: Mudit Soni

	Usage:
	# Curl
	>curl -X POST -H "Content-Type: application/json" 
        -d "{ \"likes\":\"The salary is good\", \"dislikes\":\"no job security\" }" http://127.0.0.1:5000/predict 
	
	Output:
	{
    "Compensation & Benefits":"[4]",
    "Job Security":"[1]",
    "Overall Rating":"[3]",
    "dislikes":"no job security",
    "likes":"The salary is good"
    }

    Note: Need to train classifier model on more data.
"""

import numpy as np
import flask
from tensorflow.keras.models import load_model, model_from_json
from sentence_embedder import *
from tensorflow.python.keras.backend import set_session

app = flask.Flask(__name__)
classifier= None
models= dict.fromkeys(['Overall Rating', 'Skill development/learning', 'Work-Life balance', 'Compensation & Benefits', 
                        'Company culture', 'Job Security', 'Career growth & opportunities', 'Work Satisfaction'])
embedder= SentenceEmbedder()
# Set the session and graph. Necessary to run the model on flask.
sess = tf.Session()
graph = tf.get_default_graph()

def get_models():
    """Loads all the models"""
    global classifier
    classifier= load_model('classifier.model')
    global models
    for param in list(models.keys()):
        with open("models/"+param.replace('/','-')+"_model.json", "r") as json_file:
            models[param] = model_from_json(json_file.read())
        models[param].load_weights("models/"+param.replace('/','-')+"_model.h5")

def prepare_text(text):
    """Returns embeddings of text."""
    embeddings = embedder.get(text)
    return embeddings

@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        likes= params.get("likes")
        dislikes= params.get("dislikes")
        data['likes']= likes
        data['dislikes']= dislikes
        x= np.concatenate((prepare_text(likes),prepare_text(dislikes))).reshape(-1, 1024)
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            classes= classifier.predict(x)>0.3
            #print(classes)
            for i, param in enumerate(list(models.keys())):
                if i==0:
                    data[param]= str(models[param].predict(x).argmax(axis=1)+1)
                else:
                    if classes[0,i-1]==1:
                        data[param]= str(models[param].predict(x).argmax(axis=1)+1)

    # return a response in json format 
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    set_session(sess)
    get_models()
    app.run()
