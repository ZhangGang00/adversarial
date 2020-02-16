import numpy as np
import matplotlib.pyplot as plt

def DisCo_np(X,Y):
	ajk = np.abs(np.reshape(np.repeat(X,len(X)),[len(X),len(X)]) - np.transpose(X))
	bjk = np.abs(np.reshape(np.repeat(Y,len(Y)),[len(Y),len(Y)]) - np.transpose(Y))

	Ajk = ajk - np.mean(ajk,axis=0)[None, :] - np.mean(ajk,axis=1)[:, None] + np.mean(ajk)
	Bjk = bjk - np.mean(bjk,axis=0)[None, :] - np.mean(bjk,axis=1)[:, None] + np.mean(bjk)

	dcor = np.sum(Ajk*Bjk) / np.sqrt(np.sum(Ajk*Ajk)*np.sum(Bjk*Bjk))
	return dcor


N = 1000
z0 = np.random.normal(0,1,N)
z1 = np.random.normal(0,1,N)
X = z0
xvals = []
yvals = []
for rho in np.linspace(-1,1,20):
	Y = z0*rho + z1*(1.-rho**2)**0.5
	xvals+=[rho]
	yvals+=[DisCo_np(X,Y)]

plt.plot(xvals,yvals)
plt.xlabel("Pearson Correlation")
plt.ylabel("Distance Correlation")
plt.savefig("plots/Pearson.pdf")
plt.close()



import h5py

f  = h5py.File('../../input/data_10000.h5', 'r')
plt.hist(f['dataset']['table'][:]['m'][f['dataset']['table'][:]['signal']==1],bins=np.linspace(50,150,20),alpha=0.5)
plt.hist(f['dataset']['table'][:]['m'][f['dataset']['table'][:]['signal']==0],bins=np.linspace(50,150,20),alpha=0.5)
plt.savefig("plots/histogram.pdf")
plt.close()

from keras import backend as K
import keras.layers as layers
from keras.models import Model
from keras import Sequential
from keras.layers import Lambda, Dense, Flatten
from sklearn.metrics import roc_curve, roc_auc_score
from keras.losses import binary_crossentropy
import tensorflow as tf

X = f['dataset']['table'][:]['D2'][f['dataset']['table'][:]['signal']==0]
Y = f['dataset']['table'][:]['m'][f['dataset']['table'][:]['signal']==0]
background_x = tuple(zip(X, Y))  
background_y = np.zeros(len(background_x))

X = f['dataset']['table'][:]['D2'][f['dataset']['table'][:]['signal']==1]
Y = f['dataset']['table'][:]['m'][f['dataset']['table'][:]['signal']==1]
signal_x = tuple(zip(X, Y)) 
signal_y = np.ones(len(signal_x))

X = np.concatenate([background_x,signal_x])
Y = np.concatenate([background_y,signal_y])

is_train = np.random.rand(X.shape[0])<0.85

X_train = X[is_train]
Y_train = Y[is_train]

X_val = X[~is_train]
Y_val = Y[~is_train]

plt.scatter(np.array(background_x)[:,0],np.array(background_x)[:,1],alpha=0.5)
plt.scatter(np.array(signal_x)[:,0],np.array(signal_x)[:,1],alpha=0.5)
plt.xlabel(r"$D_{2}$")
plt.ylabel("Mass [GeV]")
plt.xlim([0,10])
plt.ylim([50,150])
plt.savefig("plots/scatter.pdf")
#print(DisCo_np(np.array(background_x)[:,0],np.array(background_x)[:,1]))
#print(DisCo_np(np.array(signal_x)[:,0],np.array(signal_x)[:,1]))
plt.close()


Nepochs = 5

def redacted_set(x):
	#Returns everything except the last element.  This is the decorrelation target ("Mass")
	return x[:,0:-1]

#This network is rather generic and simple.  You can swap it out with something more 
#complicated.  Just need to use the custom first layer to separate out the mass if 
#you don't want it to be used directly in the training (aside from the decorrelation penality).
model = Sequential()
model.add(Lambda(redacted_set,input_shape =(2,)))
model.add(Dense(128, activation='relu')) 
model.add(layers.Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist_model = model.fit(X_train, Y_train, epochs=Nepochs, batch_size=128,validation_data=(X_val, Y_val))
preds = model.predict(X_val)
fpr, tpr, _ = roc_curve(Y_val, 1-preds)

plt.clf()   #clean the figure
plt.plot(tpr,fpr)
plt.plot([0,1],[0,1])
plt.xlabel("True Positive Rate")
plt.ylabel("1 - False Positive Rate")
plt.savefig("plots/ROC_pre.pdf")
plt.close()

def DisCo(y_true, y_pred, x_in, alpha = 0.):
	#alpha determines the amount of decorrelation; 0 means no decorrelation.
	#Note that the decorrelating feature is also used for learning.
	    
	X_in = tf.gather(x_in, [1], axis=1) #decorrelate with the second element of the input (=mass)
	Y_in = y_pred

	#Only require decorrelation for the background.
	mymaskX = tf.where(Y_in<1,K.ones_like(X_in),K.zeros_like(X_in))
	mymaskY = tf.where(Y_in<1,K.ones_like(Y_in),K.zeros_like(Y_in))
	print mymaskX
	#X = tf.boolean_mask(X_in, mymaskX)
	#Y = tf.boolean_mask(Y_in, mymaskY)
	X = X_in
	Y = Y_in
	    
	LX = K.shape(X)[0]
	LY = K.shape(Y)[0]
			    
	X=K.reshape(X,shape=(LX,1))
	Y=K.reshape(Y,shape=(LY,1))    
	    
	ajk = K.abs(K.reshape(K.repeat(X,LX),shape=(LX,LX)) - K.transpose(X))
	bjk = K.abs(K.reshape(K.repeat(Y,LY),shape=(LY,LY)) - K.transpose(Y))

	Ajk = ajk - K.mean(ajk,axis=0)[None, :] - K.mean(ajk,axis=1)[:, None] + K.mean(ajk)
	Bjk = bjk - K.mean(bjk,axis=0)[None, :] - K.mean(bjk,axis=1)[:, None] + K.mean(bjk)

	dcor = K.sum(Ajk*Bjk) / K.sqrt(K.sum(Ajk*Ajk)*K.sum(Bjk*Bjk))    
	    
	return binary_crossentropy(y_true,y_pred) + alpha*dcor

model = Sequential()
model.add(Dense(128, activation='relu',input_shape =(2,))) 
model.add(layers.Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=lambda y_true, y_pred: DisCo(y_true, y_pred, model.input, alpha = 10.), optimizer='adam', metrics=['accuracy'])
hist_model = model.fit(X_train, Y_train, epochs=5*Nepochs, batch_size=128,validation_data=(X_val, Y_val))
preds = model.predict(X_val)
fpr0, tpr0, _ = roc_curve(Y_val, 1-preds)

plt.plot(tpr,fpr,color='blue',label="No DisCo")
plt.plot(tpr0,fpr0,linestyle=":",color='red',label="With DisCo")
plt.plot([0,1],[0,1],color='gray')
plt.xlabel("True Positive Rate")
plt.ylabel("1 - False Positive Rate")
plt.legend(frameon=False)
plt.savefig("plots/ROC.pdf")
plt.close()
