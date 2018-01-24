import general as g
from alphabet import define_alphabet, decode_langid
from data_extractor import get_train_test_data, get_feature

import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.optimizers
from keras.utils import plot_model
import keras.utils
from keras.models import model_from_json
from keras.models import load_model as load




def createModel (input_size):
    model = Sequential()
    model.add(Dense(500,input_dim=input_size,kernel_initializer="glorot_uniform",activation="sigmoid"))
    model.add(Dropout(0.5))
    model.add(Dense(300,kernel_initializer="glorot_uniform",activation="sigmoid"))
    model.add(Dropout(0.5))
    model.add(Dense(100,kernel_initializer="glorot_uniform",activation="sigmoid"))
    model.add(Dropout(0.5))
    model.add(Dense(len(g.languages_dict),kernel_initializer="glorot_uniform",activation="softmax"))
    model_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
              optimizer=model_optimizer,
              metrics=['accuracy'])
    return model

def train_save_model(input_size, X_train, Y_train, X_test, Y_test):
    model = createModel(input_size)
    start_time = time.time()
    history = model.fit(X_train,Y_train,
          epochs= 12, #12
          validation_split=0.10,
          batch_size=64,
          verbose=2,
          shuffle=True)
    finish_time = time.time()
    print ("Learning took " + str(datetime.timedelta(seconds=finish_time - start_time)))
   
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
       json_file.write(model_json)
     # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

def main():
    alphabet = define_alphabet()
    input_size = len(get_feature('a', alphabet))

    (X_train, Y_train, X_test, Y_test) = get_train_test_data(input_size)
    #model = train_save_model(input_size, X_train, Y_train, X_test, Y_test)
    model = load_model()
    
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # now we will face the TRUTH. What is our model real accuracy tested on unseen data?
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # and now we will prepare data for scikit-learn classification report
    Y_pred = model.predict_classes(X_test)
    Y_pred = keras.utils.to_categorical(Y_pred, num_classes=len(g.languages_dict))
    # and run the report
    target_names =  list(g.languages_dict.keys())
    print(classification_report(Y_test, Y_pred, target_names=target_names))

    # show plot accuracy changes during training
"""   plt.plot(history.history['acc'],'g')
    plt.plot(history.history['val_acc'],'r')
    plt.title('accuracy across epochs')
    plt.ylabel('accuracy level')
    plt.xlabel('# epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
    plt.waitforbuttonpress()"""


if __name__ == "__main__":
    g.init()
    main()
