import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from IPython import get_ipython



def imshow_all(images, titles=None, cmap = 'gray'):
    
    #plt.ioff()
    get_ipython().run_line_magic('matplotlib', 'inline')
    num_images = len(images)
    assert num_images > 1
    
    if titles:
        assert len(titles) == num_images
    else:
        titles = ['' for _ in range(num_images)]
        
    ncols = 3
    nrows = num_images // 3 + (num_images % ncols > 0)
    height = 6  * nrows
    width = 6 * ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(width, height))
    
    for idx, (img, label) in enumerate(zip(images, titles)):
        
        axes.flat[idx].imshow(img,  cmap=cmap, interpolation='None')
        axes.flat[idx].set_title(label, fontsize = 20)
        axes.flat[idx].axis('off')
        
    for ax in axes.flat[idx + 1:]:
        ax.set_visible(False)
    
    plt.show()
        
      
        
def weighted_sparse_crossentropy(weights):
    
    
    def loss(y_true, y_pred):
        
        local_weights = tf.gather(weights, tf.argmax(y_true, axis = -1))
        ce_3d = K.sparse_categorical_crossentropy(y_true, y_pred)
        ce_3d_weighted = ce_3d * local_weights
        final = K.mean(K.mean(ce_3d_weighted, axis = (1,2)))
        return final
    return loss



def get_logits(orig_model):
    
    clone = keras.models.clone_model(orig_model)
    clone.set_weights(orig_model.get_weights()) 
        
    logits = clone.get_layer('logits').output
    model =  keras.Model(inputs = [clone.input], outputs = [logits])
    for layer in model.layers:
        layer.trainable = False
    
    return model