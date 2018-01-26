import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

import midi_manipulation

import os

#2 = Tensorflow INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs

songs = get_songs('classics')  # These songs have already been converted from midi to msgpack
print("{} songs processed".format(len(songs)))


lowest_note = midi_manipulation.lowerBound 
highest_note = midi_manipulation.upperBound
note_range = highest_note - lowest_note 


num_timesteps = 30 
n_visible = 2 * note_range * num_timesteps 
n_hidden = 120 

num_epochs = 500
batch_size = 100  # The number of training examples that we are going to send through the restricted Boltzmann machine
lr = tf.constant(0.005, tf.float32)  # learning rate

x = tf.placeholder(tf.float32, [None, n_visible], name="x") # input
W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh")) #bias for hidden
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv")) #bias for visible

#returns a random vector of 0s and 1s sampled from the input vector
def sample(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


# Propagate the visible values to sample the hidden values
def propogate_visible_to_hidden(x, W, bh):
    return sample(tf.sigmoid(tf.matmul(x, W) + bh)) 

# Propagate the hidden values to sample the visible values
def propogate_hidden_to_visible(h, W, bv):
    return sample(
            tf.sigmoid(tf.matmul(h, tf.transpose(W)) + bv))  

# This function runs a k-step gibbs chain. 
def gibbs_sample(k):
    xk = x
    for count in range (k):
      hk = propogate_visible_to_hidden(xk, W, bh)
      xk = propogate_hidden_to_visible(hk, W, bv)
      count += 1
    x_sample = xk
    return x_sample



x_sample = gibbs_sample(1)
h = propogate_visible_to_hidden(x, W, bh)
h_sample = propogate_visible_to_hidden(x_sample, W, bh)

# Next, we update the values of W, bh, and bv,
# based on the difference between the samples that we drew and the original values
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder = tf.multiply(lr / size_bt,
                      tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

# sess.run(updt) will run all 3 update steps in TensorFlow 
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]


with tf.Session() as sess:
    #training phase
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            song_len = int(np.floor(song.shape[0] // num_timesteps) * num_timesteps)
            song = song[:song_len]
            song = np.reshape(song, [song.shape[0] // num_timesteps, song.shape[1] * num_timesteps])
            # Train the RBM on batch_size examples at a time
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                sess.run(updt, feed_dict={x: tr_x})


    # Create music
    sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue
        # Here we reshape the vector to be time x notes, and then save the vector as a midi file
        S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
        midi_manipulation.noteStateMatrixToMidi(S, "out/generated_chord_{}".format(i))



