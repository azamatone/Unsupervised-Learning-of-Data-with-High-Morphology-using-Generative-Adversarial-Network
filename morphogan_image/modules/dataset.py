import math
import glob
import os

# for image io
from skimage import io
from skimage.transform import resize

import numpy as np
import tensorflow as tf


# for mnist
from tensorflow.examples.tutorials.mnist import input_data

# List all dir with specific name 
def list_dir(folder_dir, ext="png"):
    all_dir = sorted(glob.glob(folder_dir+"*."+ext), key=os.path.getmtime)
    return all_dir

def imread(path, is_grayscale=False):
    if (is_grayscale):
        img = io.imread(path, is_grayscale=True).astype(np.float)
    else:
        img = io.imread(path).astype(np.float)
    return np.array(img)  
        
# processing for galaxy dataset
def preprocess(img):
    re_size   = 128    
    img       = resize(img, [re_size, re_size])
    return img
    
class Dataset(object):

    def __init__(self, name='mnist', source='./data/mnist/', one_hot=True, batch_size = 64, seed = 0):


        self.name            = name
        self.source          = source
        self.one_hot         = one_hot
        self.batch_size      = batch_size
        self.seed            = seed
        np.random.seed(seed) # To make your "random" minibatches the same as ours

        self.count           = 0

        tf.set_random_seed(self.seed)  # Fix the random seed for randomized tensorflow operations.

        if name == 'mnist':
            self.mnist = input_data.read_data_sets(source)
            self.data  = self.mnist.train.images
            print('data shape: {}'.format(np.shape(self.data)))
            self.minibatches = self.random_mini_batches(self.data.T, self.batch_size, self.seed)
        elif name == 'galaxyzoo':
            # Count number of data images
            self.im_list  = list_dir(source, 'jpg')
            self.nb_imgs  = len(self.im_list)
            self.nb_compl_batches  = int(math.floor(self.nb_imgs/self.batch_size))
            self.nb_total_batches     = self.nb_compl_batches
            if self.nb_imgs % batch_size != 0:
                self.num_total_batches = self.nb_compl_batches + 1
            self.count = 0
            self.color_space = 'RGB'
        
    def db_name(self):
        return self.name

    def data_dim(self):
        if self.name == 'mnist':
            return 784  #28x28x1
        elif self.name == 'galaxyzoo':
            return 49152 #128x128x3
        else:
            print('data_dim is unknown.\n')

    def data_shape(self):
        if self.name == 'mnist':
            return [28, 28, 1]
        elif self.name == 'galaxyzoo':
            return [128, 128, 3]
        else:
            print('data_shape is unknown.\n')
            
    def mb_size(self):
        return self.batch_size

    def next_batch(self):

        if self.name == 'mnist': #or self.name == 'stl10':
            if self.count == len(self.minibatches):
                self.count = 0
                self.minibatches = self.random_mini_batches(self.data.T, self.batch_size, self.seed)
            batch = self.minibatches[self.count]
            self.count = self.count + 1
            return batch.T
        elif self.name == 'galaxyzoo':
            batch = self.random_mini_batches([], self.batch_size, self.seed)
            return batch

    # Random minibatches for training
    def random_mini_batches(self, X, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X)
        """
        
        if self.name == 'mnist': 
            m = X.shape[1] # number of training examples
            mini_batches = []
                
            # Step 1: Shuffle (X, Y)
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]

            # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
            num_complete_minibatches = int(math.floor(m/self.batch_size)) 
            for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[:, k * self.batch_size : (k+1) * self.batch_size]
                mini_batches.append(mini_batch_X)
            
            return mini_batches
            
        elif self.name == 'galaxyzoo':
            
            if self.count == 0:
                self.permutation = list(np.random.permutation(self.nb_imgs))
                cur_batch = self.permutation[self.count * self.batch_size : (self.count + 1) * self.batch_size]  
            elif self.count > 0 and self.count < self.nb_compl_batches:
                cur_batch = self.permutation[self.count * self.batch_size : (self.count + 1) * self.batch_size]                           
            elif self.count == self.nb_compl_batches and self.num_total_batches > self.nb_compl_batches:
                self.count = 0
                self.permutation = list(np.random.permutation(self.nb_imgs))
                cur_batch = self.permutation[self.count * self.batch_size : (self.count + 1) * self.batch_size]                
            else:
                print('something is wrong with mini-batches')
            
            mini_batches = []

            # handle complete cases
            for k in cur_batch:
                img = imread(self.im_list[k])                
                img = preprocess(img)
                img = img / 255.0                
                mini_batches.append(np.reshape(img,(1,np.shape(img)[0] * np.shape(img)[1] * np.shape(img)[2])))
            
            mini_batches = np.concatenate(mini_batches, axis=0)
            self.count = self.count + 1
                    
            return mini_batches
