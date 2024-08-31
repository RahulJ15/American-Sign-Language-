import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import numpy as np
import random
from keras.utils import to_categorical

train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

train_data=np.array(train,dtype=float)
test_data=np.array(test,dtype=float)

class_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q'
             'R','S','T','U','V','W','X','Y']

i=random.randint(1,train.shape[0])
fig1,ax1=plt.subplots(figsize=(2,2))
plt.imshow(train_data[i,1:].reshape((28,28)),cmap='gray')
print('label: ',class_names[int(train_data[i,0])])

fig=plt.figure(figsize=(18,18))
ax1=fig.add_subplot(221)
train['label'].value_counts().plot(kind='bar',ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('label') 

X_train=train_data[:,1:]/255.
X_test=test_data[:,1:]/255.

y_train=train_data[:,0]
y_train_cat=to_categorical(y_train,num_classes=25)

y_test=test_data[:,0]
y_test_cat=to_categorical(y_test,num_classes=25)

X_train=X_train.reshape(X_train.shape[0],*(28,28,1))
X_test=X_test.reshape(X_test.shape[0],*(28,28,1))

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

model = Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 25 , activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

history = model.fit(X_train,y_train_cat,batch_size=128 ,epochs = 5 , validation_data = (X_test, y_test_cat) , callbacks = [learning_rate_reduction])

prediction=model.predict(X_test)
predicted_clasess=prediction.argmax(axis=1)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predicted_clasess)
print('Accuracy Score=',accuracy)

i=random.randint(1,len(prediction))
plt.imshow(X_test[i,:,:,0])
print("Predicted Lable:",class_names[int(predicted_clasess[i])])
print("True Label:",class_names[int(y_test[i])])

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,predicted_clasess)
# Assuming you already have cm and want to plot it
plt.figure(figsize=(12, 12))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidth=.5)

plt.show()


model.save('smnist.h5')



