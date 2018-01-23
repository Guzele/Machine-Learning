import general as g
from alphabet import define_alphabet, decode_langid
import numpy as np
import os
import random
import time
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def count_chars(text,alphabet):
    alphabet_counts = []
    for letter in alphabet:
        count = text.count(letter)
        alphabet_counts.append(count)
    return alphabet_counts

def count_first (text, alphabet):
    words =  text.split(sep=' ') 
    alphabet_counts = []
    for letter in alphabet:
        alphabet_counts.append(0)
    for word in words:
        alphabet_counts [alphabet.charAt(word[0])] += 1
    return alphabet_counts

def average_word_size (text):
    words =  text.split(sep=' ') 
    return round (len (text) * 10.0 / len (words))

#get_input_row
def get_feature(sample_text, alphabet):
    counted_chars_all = count_chars(sample_text.lower(),alphabet)

    all_parts = counted_chars_all + [average_word_size (sample_text)] # + counted_chars_big
    return all_parts 


def _load_data(alphabet):

    input_size = len(get_feature('a', alphabet))
    sample_data = np.empty((g.num_lang_samples*len(g.languages_dict), input_size+1),dtype = np.uint16)
    index = 0
    for lang_code in g.languages_dict:

        path = os.path.join(g.cleaned_dir,lang_code + '.txt')
        print ("Processing file : " + path)
        f = open(path,'r')
        text = f.read()
        f.close()

        data = text.split(sep='\n') [0: g.num_lang_samples]
        for line in data:
            input_row = get_feature(line, alphabet)
            sample_data[index, ] = input_row + [g.languages_dict[lang_code]]
            index += 1
    np.random.shuffle(sample_data)
    return sample_data


def save_dataset(alphabet):
    alphabet = define_alphabet()
    input_size = len(get_feature('a', alphabet))
    sample_data = _load_data(alphabet)
    print (100*"-")
    print ("Samples array size : ",sample_data.shape )
    path_smpl = os.path.join(g.samples_dir,str(input_size)+".npz")
    np.savez_compressed(path_smpl,data=sample_data)
    print("Dataset saved to " ,path_smpl)
    del sample_data
def load_dataset(alphabet):
     input_size = len(get_feature('a', alphabet))
     path_smpl = os.path.join(g.samples_dir, str(input_size)+".npz")
     dt = np.load(path_smpl)['data']
     return dt

def get_train_test_data(input_size):
    path_tt = os.path.join(g.train_test_dir,str(input_size)+".npz")
    train_test_data = np.load(path_tt)
    X_train = train_test_data['X_train']
    print ("X_train: ",X_train.shape)
    Y_train = train_test_data['Y_train']
    print ("Y_train: ",Y_train.shape)
    X_test = train_test_data['X_test']
    print ("X_test: ",X_test.shape)
    Y_test = train_test_data['Y_test']
    print ("Y_test: ",Y_test.shape)
    #del train_test_data
    return (X_train, Y_train, X_test, Y_test)

def save_train_test_data(input_size, dt):
    dt = dt.astype(np.float64)

    # X and Y split
    X = dt[:,0:input_size]
    Y = dt[:,input_size]
    del dt

    # random index to check random sample
    random_index = random.randrange(0,X.shape[0])
    print("Example data before processing:")
    print("X : \n", X[random_index,])
    print("Y : \n", Y[random_index])
    time.sleep(120) # sleep time to allow release memory. This step is very memory consuming
    # X preprocessing
    # standar scaler will be useful laterm during DNN prediction
    standard_scaler = preprocessing.StandardScaler().fit(X)
    X = standard_scaler.transform(X)   
    print ("X preprocessed shape :", X.shape)
    # Y one-hot encoding
    Y = to_categorical(Y, num_classes=len(g.languages_dict))
    # See the sample data
    print("Example data after processing:")
    print("X : \n", X[random_index,])
    print("Y : \n", Y[random_index])
    # train/test split. Static seed to have comparable results for different runs
    seed = 42
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
    del X, Y
    # wait for memory release again
    time.sleep(120)
    # save train/test arrays to file
    path_tt = os.path.join(g.train_test_dir, str(input_size)+".npz")
    np.savez_compressed(path_tt,X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
    print("Train and test data saved to ", path_tt)
    del X_train,Y_train,X_test,Y_test

def main():
    alphabet = define_alphabet()
    input_size = len(get_feature('a', alphabet))
    
    save_dataset(alphabet)
    
    dt = load_dataset(alphabet)
    random_index = random.randrange(0,dt.shape[0])
    print ("Sample record : \n",dt[random_index,])
    print ("Sample language : ",decode_langid(dt[random_index,][input_size]))
    # we can also check if the data have equal share of different languages
    print ("Dataset shape :", dt.shape)
    bins = np.bincount(dt[:,input_size])
    print ("Language bins count : ") 
    for lang_code in g.languages_dict: 
        print (lang_code,bins[g.languages_dict[lang_code]])

    save_train_test_data(input_size, dt)    



if __name__ == "__main__":
    g.init()
    main()
