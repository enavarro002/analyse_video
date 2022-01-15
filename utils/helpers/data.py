from keras.applications.vgg16 import preprocess_input
from cv2 import cvtColor, COLOR_BGR2RGB, resize
import utils.helpers.constants as path

def preprocess_image(img) : 
    
    img_rgb = cvtColor(img, COLOR_BGR2RGB)
    img_resize = resize(img_rgb, dsize=(227,227))
    return preprocess_input(img_resize)

    
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical
def get_datasets(train=True, val=True, test=True):
    generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    resp = []

    if train:
        ## Train dataset
        print("Train:\t\t", end="")
        train_img = generator.flow_from_directory(
            directory=path.__train_path__, target_size=(227, 227), color_mode='rgb', classes=None,
            class_mode='categorical', batch_size=32, shuffle=True, seed=None,
            save_to_dir=None, save_prefix='', save_format='png',
            follow_links=False, interpolation='nearest'
        )
        assert train_img.num_classes == 5
        train_labels = to_categorical(train_img.classes)        
        resp.append(train_img)
        resp.append(train_labels)

    if val:
        ## Validation dataset
        print("Validation:\t", end="")
        val_img = generator.flow_from_directory(
            directory=path.__val_path__, target_size=(227, 227), color_mode='rgb', classes=None,
            class_mode='categorical', batch_size=32, shuffle=True, seed=None,
            save_to_dir=None, save_prefix='', save_format='png',
            follow_links=False, interpolation='nearest'
        )
        assert val_img.num_classes == 5
        val_labels = to_categorical(val_img.classes)        
        resp.append(val_img)
        resp.append(val_labels)

    if test:
        ## Test dataset
        print("Test:\t\t", end="")
        test_img = generator.flow_from_directory(
            directory=path.__test_path__, target_size=(227, 227), color_mode='rgb', classes=None,
            class_mode='categorical', batch_size=32, shuffle=False, seed=None,
            save_to_dir=None, save_prefix='', save_format='png',
            follow_links=False, subset=None, interpolation='nearest'
        )
        assert test_img.num_classes == 5
        test_labels = to_categorical(test_img.classes)
        resp.append(test_img)
        resp.append(test_labels)
    
    return resp

import os
def get_labels():
    class_labels = os.listdir(path.__train_path__)
    return class_labels, len(class_labels)

from random import choice
from cv2 import imread
import numpy as np
def get_random_test_img(nb_img):
    img_test = []
    for _ in range(nb_img):
        folder_name = choice(os.listdir(path.__test_path__))
        file_name = choice(os.listdir(os.path.join(path.__test_path__,folder_name)))
        pth = os.path.join(path.__test_path__,folder_name, file_name)
        img = imread(pth)
        img_test.append(img)
    
    return np.array(img_test)
