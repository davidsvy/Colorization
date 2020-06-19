import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from config import MAX_ABS_VALUE, IMG_SHAPE, TRAIN_PATH, QUANTIZE_DIST, QUANTIZE_THRESHOLD



def get_color_freqs(path, img_shape, max_imgs):
    
    def rgb2lab(rgb_img):
    
        return cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2LAB)
    
    datagen = ImageDataGenerator(
        preprocessing_function = rgb2lab)

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
        num_imgs = min(generator.samples, max_imgs)
        
    color_freqs = np.zeros((MAX_ABS_VALUE, MAX_ABS_VALUE))
    
    
    for _ in tqdm(range(num_imgs)):
        
        img_lab = next(generator)[0, ...].astype(np.uint8)
        img_ab = np.reshape(img_lab[..., 1:], (-1, 2))
        color_freqs[img_ab[:, 0], img_ab[:, 1]] += 1
        
    
    np.save('data/color_freqs.npy', color_freqs)
   
 
    
    
def quantize_colors(dist, threshold):
    
    assert 0.0 < threshold < 1.0
    
    color_freqs = np.load('data/color_freqs.npy')

    lab_quantized = np.zeros((MAX_ABS_VALUE, MAX_ABS_VALUE, 3), dtype = np.uint8)
    lab_quantized[..., 0] = MAX_ABS_VALUE - 1
    lab_quantized[..., 1] = MAX_ABS_VALUE // 2
    lab_quantized[..., 2] = MAX_ABS_VALUE // 2
    
    color_freqs_quantified = np.zeros((color_freqs.shape[0] // dist, color_freqs.shape[1] // dist))
    color_centres = []
    color_weights = []
    num_total = dist ** 2
    for i in range(0, color_freqs.shape[0] - dist, dist):
        for j in range(0, color_freqs.shape[1] - dist, dist):
            freq_sum = np.sum(color_freqs[i:i+dist, j:j+dist])
            num_nonempty = np.sum(color_freqs[i:i+dist, j:j+dist] > 0)
            if num_nonempty / num_total > threshold:
                
                square_centre = (i + dist // 2 , j + dist // 2 )
                color_centres.append(square_centre)
                lab_quantized[i : i+dist, j : j+dist, 1:] = square_centre
                lab_quantized[i : i+dist, j : j+dist, 0] = 169
                color_freqs_quantified[i // dist, j // dist] = freq_sum
                color_weights.append(freq_sum)
    
    rgb_quantized = cv2.cvtColor(lab_quantized, cv2.COLOR_LAB2RGB)
    color_centres = np.array(color_centres)
    color_weights = np.array(color_weights)

    
    eps = 1e-5
    T = 1.6
    temp = np.exp(np.log(1 +color_weights) / T)
    probs_smoothed = temp / np.sum(temp)
    weights_smoothed = 1 / (eps + probs_smoothed) ** 0.6
    norm_coef = np.sum(weights_smoothed * probs_smoothed)
    weights_final = weights_smoothed / norm_coef
    
    #weights_tf = tf.convert_to_tensor(weights_smoothed, dtype = tf.float32)
    np.save('data/weights_final.npy', weights_final)
    np.save('data/color_centres.npy', color_centres)
    np.save('data/color_weights.npy', color_weights)
        
    plt.ioff()
    plt.imsave('data/quantized_rgb.png', rgb_quantized)
    
    plt.figure()
    plt.axis('off')
    plt.grid(False)
    plt.imshow(np.log(1 + color_freqs), cmap = 'plasma')
    plt.savefig('data/log_color_freqs.png', bbox_inches="tight", transparent=True)
    
    plt.figure()
    plt.axis('off')
    plt.grid(False)
    plt.imshow(np.log(1 + color_freqs_quantified), cmap = 'plasma')
    plt.savefig('data/quantized_log_color_freqs.png', bbox_inches="tight", transparent=True)
    
    

    
if not os.path.exists('data/color_freqs.npy'):    
    get_color_freqs(TRAIN_PATH, IMG_SHAPE, -1)
    
quantize_colors(QUANTIZE_DIST, QUANTIZE_THRESHOLD)
       