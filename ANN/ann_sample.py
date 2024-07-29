
################################# Artificial Neural Network #################################


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]
X
X.shape
y.shape


#Create dummy variables - one hot encoding for nominal data
geography=pd.get_dummies(X["Geography"],drop_first=True)  
gender=pd.get_dummies(X['Gender'],drop_first=True)

geography
gender

## Concatenate the Data Frames
X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Part 2 - Now let's make the ANN!
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential   # sequential library is responsible in creating neural network
from keras.layers import Dense        # for hidden layers
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout      # used for regularization 

'''
When to use a Sequential model
A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor 
and one output tensor.    A tensor is a multidimentional array'''

# Initialising the ANN
classifier = Sequential()


#### All hidden layers can have either relu or leaky relu to address the vanishing gradient issue

# Adding the input layer and the first hidden layer
''' here we are creating a neural network with 11 input neurons(features of X train) and 6 neurons in 1st hidden layers and 
we use he_usinform for Relus activation function.......dense function is used to create neurons'''
classifier.add(Dense(units = 6, kernel_initializer= 'he_uniform',activation='relu',input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(units = 1,  kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))  # 1 because of binary classification

classifier.summary()


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10,epochs = 100)
 

# list all data in history

print(model_history.history.keys())

model.layers[0].get_weights()
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  #creating a threshold for output .. if more than 0.5 the yes else no

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics
cm = confusion_matrix(y_test, y_pred)
cm

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
cm_display.plot()
plt.show()

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
score


