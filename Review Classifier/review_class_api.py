"""
	Usage:
	# Browser 
	http://127.0.0.1:5000/predict?text=The_instructor_is_funny

	# Curl
	>curl -X POST -H "Content-Type: application/json" -d "{ \"text\":\"The instructor is funny\" }" http://127.0.0.1:5000/predict
	
	Output:
	{
	"is_difficulty":"False",
	"is_doubt_resolution":"False",
	"is_instructor":"True",
	"is_notes_slides":"False",
	"is_student_discussion":"False",
	"is_sylabus":"False",
	"is_test":"False",
	"text":"The instructor is funny"
	}
"""

import numpy as np
import flask
from tensorflow.keras.models import load_model, model_from_json
from sentence_embedder import *
from tensorflow.python.keras.backend import set_session

app = flask.Flask(__name__)
model= None
#embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embedder= SentenceEmbedder()
# Set the session and graph. Necessary to run the model on flask.
sess = tf.Session()
graph = tf.get_default_graph()

def get_model():
	"""Loads the model"""
    global model
    json_file = open('review_class.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("review_class.h5")

def prepare_text(text):
	"""Returns embeddings of text."""
    embeddings = embedder.get(text)
    return embeddings

@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {'text': '0', 'is_instructor': '0', 'is_difficulty': '0', 'is_sylabus': '0',
       		'is_test': '0', 'is_notes_slides': '0', 'is_doubt_resolution': '0', 'is_student_discussion': '0'}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=params.get("text")
        data['text']= x
        x= prepare_text(x)
        global sess
        global graph
        with graph.as_default():
        	set_session(sess)
        	preds= model.predict(x.reshape((-1,512)))
        for i, key in enumerate(list(data.keys())[1:]):
        	data[key]= str(preds[0,i]>0.5)

    # return a response in json format 
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    set_session(sess)
    get_model()
    app.run()