import pickle
import os
import numpy as np


_dataset_path = '../dataset/cifar-10-batches-py/'
labels = ['airplane', 'automobile', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck']


def _preprocess(image):
    '''this methods normalizes the images'''
    image = image/255.0
    return image


def _get_batches(dataset_path):
    files = os.listdir(dataset_path)
    batches = [f for f in files if 'data_batch_' in f]
    return batches


def _convert_to_array(batch_path):
    '''here we read the pickle file and extract the images and labels and convert them into numpy array format'''
    with open(batch_path, 'rb') as batch:
        batch_data = pickle.load(batch, encoding='bytes')
        images = batch_data[b'data']
        images = images.reshape((-1, 3, 32, 32))
        images = images.transpose([0, 3, 2, 1])
        labels = np.array(batch_data[b'labels'])
        return images, labels


def _shuffle_data(x, y):
    perm = np.random.permutation(len(x))
    return x[perm], y[perm]


def load_data(limit=2, shuffle=False, dataset_path=_dataset_path):
    '''Load the CIFAR-10 dataset'''
    batches = _get_batches(dataset_path)[:limit]
    x = []
    y = []
    for batch in batches:
        batch_path = os.path.join(dataset_path, batch)
        temp_x, temp_y = _convert_to_array(batch_path)
        temp_x = np.vectorize(_preprocess)(temp_x)
        x.extend(temp_x)
        y.extend(temp_y)
    x = np.array(x)
    y = np.array(y)
    if shuffle:
        x, y = _shuffle_data(x, y)
    return x[:-100], y[:-100], x[-100:], y[-100:]