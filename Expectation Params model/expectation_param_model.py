"""
   Author: Mudit Soni
"""
import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K
from keras.callbacks import EarlyStopping
#from sentence_embedder import *
from custom_losses import *
import pickle

np.random.seed(31415)

def oversample(x1, x2, y):
	"""For oversampling imbalanced data."""
	nums= np.sum(y, axis=0)
	max_num= np.max(nums)
	for i in range(nums.shape[0]):
		diff= max_num- nums[i]
		yp= y[y[:,i]==1]
		x1p= x1[y[:,i]==1]
		x2p= x2[y[:,i]==1]
		choices = np.random.choice(np.arange(yp.shape[0]), int(diff))
		x1= np.concatenate([x1, x1p[choices]], axis=0)
		x2= np.concatenate([x2, x2p[choices]], axis=0)
		y= np.concatenate([y, yp[choices]], axis=0)

	return x1, x2, y

def pre_process_data(classifier, sample= None, load= False, pkl= False, test_size=0):
	"""Converts data to a trainable form for ordinal regression.

    Args:
      classifier: Keras model for classifying data into relevant expectation parameters.
      sample: A dictionary with company names as keys reviews/ratings as values.
      Load: Boolean for loading already pickeled data.
      pkl: Boolean for pickling the preprocessed data.
      test_size: A fraction giving size of test data.

    Returns:
      X_train: Training data.
      X_test: Test data.
      X_txt_train: Text for training data.
      X_txt_test: Text for test data.
      y_train: Targets for training data.
      y_test: Targets for test data.

      If test_size=0, it returns X_train, X_txt_train and y_train only.
    """
	if load==True:
		with open('x_text', 'rb') as f:
			X_txt= pickle.load(f)
		with open('train_x', 'rb') as f:
			X= pickle.load(f)
		with open('train_y', 'rb') as f:
			y_dict_mod= pickle.load(f)
	else:
		embedder= SentenceEmbedder()

		likes_list_txt= []
		dislikes_list_txt= []  # For storing text.
		likes_list= []
		dislikes_list= []  # For storing embeddings.
		y_dict= {'Overall Rating': [], 'Skill development/learning': [], 'Work-Life balance': [], 
				'Compensation & Benefits': [], 'Company culture': [], 'Job Security': [], 
				'Career growth & opportunities': [], 'Work Satisfaction': []}

		# Store the reviews and ratings
		for company in sample.keys():
			for i in range(sample[company]['Reviews'].shape[0]):
				l= sample[company]['Reviews'].loc[i, 'Likes']
				d= sample[company]['Reviews'].loc[i, 'Dislikes']
				if type(l)!=str or type(d)!=str:
					continue
				for key in y_dict.keys():
					try:  # Raises TypeError when no rating is given.
						y_dict[key].append(int(sample[company]['Reviews'].loc[i, key]))
					except:
						y_dict[key].append(0)  # Store zero if no rating is given.
				likes_list.append(embedder.get(l))
				dislikes_list.append(embedder.get(d))
				likes_list_txt.append(l)
				dislikes_list_txt.append(d)

		y_dict_mod= {}
		# Convert labels to onehot encodings.
		for key in list(y_dict.keys()):
			y_dict_mod[key]= np.zeros((len(y_dict[key]), 6))
			y_dict_mod[key][np.arange(len(y_dict[key])),y_dict[key]]=1

		temp= np.asarray(likes_list+dislikes_list)
		temp2= np.asarray(likes_list_txt+dislikes_list_txt)
		X= np.concatenate((temp[:temp.shape[0]//2],temp[temp.shape[0]//2:]), axis=1)
		X_txt= np.vstack((temp2[:temp2.shape[0]//2],temp2[temp2.shape[0]//2:])).T

		if pkl==True:  # Pickle the dataset.
			with open('x_text','wb') as f:
				pickle.dump(X_txt, f)
			with open('train_x', 'wb') as f:
				pickle.dump(X, f)
			with open('train_y', 'wb') as f:
				pickle.dump(y_dict_mod, f)

	X_train, X_test, y_train, y_test, X_txt_train, X_txt_test= {}, {}, {}, {}, {}, {}
	classes= classifier.predict(X)>0.5

	for i, param in enumerate(list(y_dict_mod.keys())):
		keep= y_dict_mod[param][:,0]==0
		if i!=0:  # We don't classify for key='Overall Rating'.
			temp= classes[:,i-1]==1
			keep[temp==False]= False

		y= y_dict_mod[param]
		y=y[keep][:,1:]  # We ignore the 0th column as ratings start from 1.
		x= X[keep]
		x_txt= X_txt[keep]

		if test_size==0:
			X_train[param], X_txt_train[param], y_train[param]= oversample(x, x_txt, y)
		else:	
			X_train[param], X_test[param], y_train[param], y_test[param], X_txt_train[param], X_txt_test[param] = train_test_split(x, y, x_txt, test_size=test_size)
			X_train[param], X_txt_train[param], y_train[param]= oversample(X_train[param], X_txt_train[param], y_train[param])

	if test_size==0:
		return X_train, y_train, X_txt_train
	else:
		return X_train, X_test, y_train, y_test, X_txt_train, X_txt_test

def pre_process_classifier_data(class_data, test_size= 0):
	"""Converts data to a trainable form for ordinal regression.

    Args:
      class: A DataFrame containing review and classes.
      test_size: A fraction giving size of test data.

    Returns:
      x_samp_tr: Training data.
      x_samp_te: Targets for training data.
      y_samp_tr: Test data.
      y_samp_te: Targets for test data.

      If test_size=0, it returns X_samp and y_samp only.
    """
	embedder= SentenceEmbedder()
	class_data= class_data.loc[class_data.Sum>0]
	X_samp= np.column_stack((np.array(class_data['0'].apply(embedder.get).to_list()),np.array(class_data['1'].apply(embedder.get).to_list())))
	y_samp= np.array(class_data.loc[:, 'Skill development/learning':'Work Satisfaction'])
	nans= np.isnan(y_samp)
	y_samp[nans]= 0  # Replace nans with zero.

	if test_size==0:
		return (X_samp, y_samp)
	else:
		x_samp_tr, x_samp_te, y_samp_tr, y_samp_te= train_test_split(X_samp, y_samp, test_size= test_size)
		return x_samp_tr, x_samp_te, y_samp_tr, y_samp_te

def classifier_model():
	"""Returns the compiled classifier model."""
	inp = Input(shape=(1024,))
	x = Dense(128, activation='relu')(inp)
	outp = Dense(7, activation='sigmoid')(x)
	model = Model(inputs=inp, outputs=outp)
	model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy'])
	return model

def regression_model():
	"""Returns the regression model for review-rating prediction."""
	inp = Input(shape=(1024,))
	x = Dense(128, activation='relu')(inp)
	outp = Dense(5, activation='softmax')(x)
	model = Model(inputs=inp, outputs=outp)
	model.compile(optimizer=tf.keras.optimizers.Nadam(), loss=CohenKappaLoss(5), 
                  metrics=[CohenKappa(num_classes=5, weightage='quadratic'),f1_m])
	return model

def train_classifier_model(X1_train, y1_train, X1_test=None, y1_test=None, model=None):
	"""Trains/retrains the model on input data"""
	# Get the classifier model.
	if model==None:
		model= classifier_model()

	earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, mode='min', restore_best_weights=True)  # For early stopping training in case there is no improvement.

	# Train the model.
	if y1_test==None:
		model.fit(X1_train, y1_train, callbacks=[earlystop], batch_size= 16, epochs= 100, verbose=0)
	else:
		model.fit(X1_train, y1_train, validation_data= (X1_test, y1_test), callbacks=[earlystop], batch_size= 16, epochs= 100, verbose=0)
	return model

def train_regression_models(X_train, y_train, params= None, X_test=None, y_test=None, models=None):
	"""Trains/retrains the models on input data

	Args:
	  X_train: Training data.
      X_test: Test data.
      y_train: Targets for training data.
      y_test: Targets for test data.
      params: A list of parameters whose models are to be trained. If "None", models are trained for all parameters.
      models: A dictionary of pretrained models for retraining.

    Returns:
      models: A dictionary of trained models
	"""
	if params== None:
		params= list(y_train.keys())
	# Get the models.
	if models==None:
		models= dict.fromkeys(params)
		for param in params:
			models[param] = regression_model()

	earlystop = EarlyStopping(monitor = 'cohen_kappa', min_delta = 0, patience = 6, mode='max', restore_best_weights=True)  # For early stopping training in case there is no improvement.

	# Train the models.
	if y_test==None:
		for param in params:
			models[param].fit(X_train[param], y_train[param], callbacks=[earlystop], batch_size=32, epochs=50, verbose=0)
	else:
		for param in params:
			models[param].fit(X_train[param], y_train[param], validation_data=(X_test[param],y_test[param]), callbacks=[earlystop], batch_size=32, epochs=50, verbose=0)
	
	return models

def test_models(X_test, y_test, models):
	"""Tests the model on given test data and returns the confusion matrix for each param"""
	params= list(y_test.keys())
	cms= dict.fromkeys(params)  # Create a dictionary for confusion matrices.

	for param in params:
		y_pred = models[param].predict(X_test[param])
		list(y_pred)
		cms[param] = confusion_matrix(y_test[param].argmax(axis=1), y_pred.argmax(axis=1))

	return cms 

# Get classification data.
#samp_data= pd.read_csv('samp_texts2.csv')  
#x_samp_tr, x_samp_te, y_samp_tr, y_samp_te= pre_process_classifier_data(samp_data, 0.1)
#x_samp_tr, y_samp_tr= pre_process_classifier_data(samp_data)

# Train classifier model.
#classifier= train_classifier_model(x_samp_tr, y_samp_tr)  

# Save/Load the classifier model.
#classifier.save('classifier.model')
classifier= load_model('classifier.model')

# Get regression data.
#with open('sample_companies.pkl', 'rb') as f:
#	sample= pickle.load(f)

X_train, X_test, y_train, y_test, X_txt_train, X_txt_test= pre_process_data(classifier, load= True, test_size=0.1)
#X_train, y_train, X_txt_train= pre_process_data(classifier, load= True)

# Train regression models.
#models= train_regression_models(X_train, y_train)

# Load models.
models= dict.fromkeys(list(y_train.keys()))
for param in list(models.keys()):
	with open("models/"+param.replace('/','-')+"_model.json", "r") as json_file:
		models[param] = model_from_json(json_file.read())
	models[param].load_weights("models/"+param.replace('/','-')+"_model.h5")

# Test models.
cms= test_models(X_test, y_test, models)
for param in list(cms.keys()):
	print(param, cms[param])

# Save models.
#for param in list(models.keys()):
#	with open("models/"+param.replace('/','-')+"_model.json", "w") as json_file:
#		json_file.write(models[param].to_json())
#	models[param].save_weights("models/"+param.replace('/','-')+"_model.h5")

