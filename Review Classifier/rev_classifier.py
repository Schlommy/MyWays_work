"""
    Author: Mudit Soni
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import multilabel_confusion_matrix
from sentence_embedder import *
from tensorflow.keras.models import load_model, Model, model_from_json
from tensorflow.keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

np.random.seed(31415)

def pre_process_data(data, test_size=0):
	"""Converts data to a trainable form.

    Args:
      data: A pandas dataframe.
      test_size: A fraction giving size of test data.

    Returns:
      X_train: Training data.
      X_test: Targets for training data.
      y_train: Test data.
      y_test: Targets for test data.

      If test_size=0, it returns X_train and y_train only.
    """
    #module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
  	#embed = hub.load(module_url)

	embedder= SentenceEmbedder() 

	data= data.loc[data1['is_test'].notnull()]  # Remove unlabeled data points.
	X= np.array(data['Review'].apply(embedder.get).to_list())
	y= np.array(data.loc[:, 'is_instructor':])

	# Replace nan values in labels with 0.
	nans= np.isnan(y)  
	y[nans]= 0

	if test_size==0:
		return (X, y)
	else:
		# Get train test split.
		X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=test_size)
		return (X_train, X_test, y_train, y_test)

def calculating_class_weights(y_true):
	"""Get class weights for imbalanced label data."""
	number_dim = y_true.shape[1]
	weights = np.empty([number_dim, 2])
	for i in range(number_dim):
		weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
	return weights

def get_weighted_loss(weights):
	"""Custom keras loss function for getting weighted binary-crossentropy loss."""
	# The actual loss function is wrapped in another function because keras only allows loss function with two arguments.
	def weighted_loss(y_true, y_pred):
		return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
	return weighted_loss

def weighted_classifier_model(weights):
	"""The keras classifier model built using functional api.

	Args:
      weights: Weights used in loss function for imbalanced data. 

    Returns:
      model: The final compiled model.
    """
	inp = Input(shape=(512,))
	x = Dense(64, activation='relu')(inp)
	outp = Dense(7, activation='sigmoid')(x)
	model = Model(inputs=inp, outputs=outp)
	model.compile(optimizer='adam', loss= get_weighted_loss(weights), 
					metrics=['accuracy'])
	return model

def train_model(X1_train, y1_train, X1_test=None, y1_test=None, model=None):
	"""Trains/retrains the model on input data"""
	# Get the classifier model.
	if model==None:
		wts= calculating_class_weights(y1_train)
		model= weighted_classifier_model(wts)

	earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, mode='min', restore_best_weights=True)  # For early stopping training in case there is no improvement.

	# Train the model.
	if y1_test==None:
		model.fit(X1_train, y1_train, callbacks=[earlystop], batch_size= 32, epochs= 1000, verbose=0)
	else:
		model.fit(X1_train, y1_train, validation_data= (X1_test, y1_test), callbacks=[earlystop], batch_size= 32, epochs= 1000, verbose=0)
	return model

def test_model(X1_test, y1_test, model):
	"""Tests the model on given test data and returns the confusion matrix"""
	y1_pred = model.predict(X1_test)  # Get model predictions on test data.
	y1_pred =(y1_pred>0.5)
	list(y1_pred)

	cm = multilabel_confusion_matrix(y1_test, y1_pred)
	return cm 

# Read data from 'reviews.csv'.
data1= pd.read_csv('reviews.csv', encoding ='latin1')
X1_train, X1_test, y1_train, y1_test= pre_process_data(data1, test_size=0.1)
#X1_train, y1_train= pre_process_data(data1, test_size=0)

# Train model.
#model= train_model(X1_train, y1_train, X1_test, y1_test)
#model= train_model(X1_train, y1_train)

# Load pretrained model.
#model= load_model(review_class.model)
json_file = open('review_class.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("review_class.h5")

# Test model.
cm= test_model(X1_test, y1_test, model)
print(cm) 

# Save model.
#model.save('review_class.model')
#with open("review_class.json", "w") as json_file:
#    json_file.write(model.to_json())
#model.save_weights("review_class.h5")

