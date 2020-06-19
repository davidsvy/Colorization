import numpy as np
from multiprocessing import cpu_count
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
from config import TRAIN_PATH, IMG_SHAPE, BATCH_SIZE
from utils import imshow_all
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


NUM_IMGS = ImageDataGenerator().flow_from_directory(TRAIN_PATH, target_size = IMG_SHAPE).samples
print('Dataset contains {} images'.format(NUM_IMGS))
STEPS_PER_EPOCH = NUM_IMGS // BATCH_SIZE
CONV_IMG_SHAPE = IMG_SHAPE + (1,)
workers = cpu_count()

color_centres = np.load('data/color_centres.npy')
Q = color_centres.shape[0]

knn_sklearn = NearestNeighbors(n_neighbors=1, n_jobs=workers)
knn_sklearn.fit(color_centres)






def custom_generator(train_path, img_shape, batch_size, workers):
    
    
    def preprocess_input(img_rgb):
    
        img_rgb = img_rgb.astype(np.uint8)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        return img_lab.astype(np.uint8)
    
    
    datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        zoom_range = [0.70, 1.0],
        horizontal_flip = True,
        rotation_range = 35,
        shear_range = 0.2,
        brightness_range = (0.9, 1.6))

    train_generator = datagen.flow_from_directory(
        train_path,
        target_size = img_shape,
        batch_size = batch_size,
        class_mode = None,
        shuffle = True
    )
    
    ground_truth_shape = (-1,) + img_shape + (1,)

    while True:
        
        img_lab = train_generator.next()
        ab_dim = np.reshape(img_lab[..., 1:].astype(np.uint8), (-1, 2))
        
        nn = knn_sklearn.kneighbors(ab_dim, return_distance=False).astype(np.uint8)
        
        ground_truth = np.reshape(nn, ground_truth_shape)
        
        model_input = np.expand_dims(img_lab[..., 0] / 255.0, axis = -1).astype(np.float32)
        
        
        yield model_input, ground_truth
        
        



def reconstruct_batch(model_input, ground_truth):
    
    res_shape = model_input.shape[:-1] + (3,)
    res = np.zeros(res_shape).astype(np.uint8)
    
    model_input = np.rint(model_input.astype(np.float32) * 255).astype(np.uint8)
    
    ground_truth = ground_truth[..., 0]
    ab = np.rint(color_centres[ground_truth]).astype(np.uint8)
    lab = np.concatenate((model_input, ab), axis = -1)
    
    for idx in range(lab.shape[0]):
      
        rgb = cv2.cvtColor(lab[idx, ...] , cv2.COLOR_LAB2RGB)
        res[idx, ...] = np.copy(rgb)

    imshow_all(res)

         
        
generator = custom_generator(
                train_path = TRAIN_PATH,
                img_shape = IMG_SHAPE,
                batch_size = BATCH_SIZE,
                workers = workers)


if __name__ == '__main__':
    
    data = next(generator)
    reconstruct_batch(data[0], data[1])

