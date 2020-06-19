import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tqdm import tqdm
import cv2
import scipy as sp
from config import Temp, TEST_PATH, IMG_SHAPE, MODEL_PATH
from utils import get_logits, imshow_all, weighted_sparse_crossentropy




def colorize(path, img_shape, model, max_imgs, Temp = 0.38, true_gray = False):
    
    
    color_centres = np.load('data/color_centres.npy')
    
    datagen = ImageDataGenerator()

    generator = datagen.flow_from_directory(
        path,
        target_size = img_shape,
        batch_size = 1,
        class_mode = None,
        shuffle = True
    )
    
    if max_imgs <= 0:
        num_imgs = generator.samples
    else:
        num_imgs = min(max_imgs, generator.samples)
        
    if num_imgs == 0:
        print('No images in given directory')
        return
        
    if num_imgs > 200:
        print('Too many images!!!')
        return
    
    out_shape = (num_imgs,) + IMG_SHAPE + (3, )
    in_shape = (num_imgs,) + IMG_SHAPE + (1, )
        
    orig_imgs = np.zeros(out_shape, dtype = np.uint8)
    pred_imgs = np.zeros(out_shape, dtype = np.uint8)
    model_input = np.zeros(in_shape, dtype = np.float32)
    
    
    for idx in tqdm(range(num_imgs)):
        
        rgb_img = generator.next()[0, ...].astype(np.uint8)
        orig_imgs[idx,...] = rgb_img
        if true_gray:
            gray_img = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray_img = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0]
        
        gray_img = gray_img.astype(np.float32) / 255.0
        gray_img = np.expand_dims(gray_img, axis = -1)
        model_input[idx, ...] = gray_img
    
        
        
    model_output = model.predict(model_input)
    temp_softmax = sp.special.softmax(model_output / Temp, axis = -1)
    a_dim = temp_softmax.dot(color_centres[..., 0])
    a_dim = np.expand_dims(a_dim, axis = -1)
    a_dim = np.rint(a_dim).astype(np.uint8)
    b_dim = temp_softmax.dot(color_centres[..., 1])
    b_dim = np.expand_dims(b_dim, axis = -1)
    b_dim = np.rint(b_dim).astype(np.uint8)
    
    
    l_dim = np.rint(model_input * 255).astype(np.uint8)
    lab_imgs = np.concatenate((l_dim, a_dim, b_dim), axis = -1)
    
    for idx, lab_img in enumerate(lab_imgs):
        
        rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
        pred_imgs[idx, ...] = rgb_img
    
    print('waiting for plt...')
    print(pred_imgs.shape)
    
    for orig_img, pred_img in zip(orig_imgs, pred_imgs):
        
        imshow_all([orig_img, pred_img],
                  ['Original image', 'Colorized image'])
        
weights = np.load('data/color_weights.npy')
weights_tf = tf.convert_to_tensor(weights, dtype = tf.float32)

      
orig_model = keras.models.load_model(MODEL_PATH, 
                   custom_objects={'loss': weighted_sparse_crossentropy(weights_tf)})

logits = get_logits(orig_model)

colorize(path = TEST_PATH, 
          img_shape = IMG_SHAPE, 
          model = logits, 
          max_imgs = 100,
          Temp = Temp,
          true_gray = False)    