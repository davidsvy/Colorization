import os
import tensorflow as tf
from config import EPOCHS, patience
from data_generator import generator, STEPS_PER_EPOCH
from model import model

if __name__ == '__main__':
    
    checkpoint_filepath = 'data/checkpoints/model_{epoch:02d}.h5'
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='train_loss',
        mode='auto',
        save_best_only=False,
        verbose = 1,
        period = 2)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor= 'train_loss', 
                                                     factor = 0.2,
                                                     patience = patience, 
                                                     verbose = 1)
    
    
    model.fit(
        generator,
        epochs = EPOCHS,
        verbose = 1,
        steps_per_epoch = STEPS_PER_EPOCH,
        callbacks=[checkpoint, reduce_lr])