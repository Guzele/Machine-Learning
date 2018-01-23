import general as g
from alphabet import define_alphabet, decode_langid
from data_extractor import get_train_test_data, get_feature


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

def main():
    alphabet = define_alphabet()
    input_size = len(get_feature('a', alphabet))

    (X_train, Y_train, X_test, Y_test) = get_train_test_data(input_size)
    model = createModel(input_size)
    history = model.fit(X_train,Y_train,
          epochs= 12, #12
          validation_split=0.10,
          batch_size=64,
          verbose=2,
          shuffle=True)
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
