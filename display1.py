import tensorflow as tf
import numpy as np
import cv2
import time

def getFrame():
    changes = np.random.uniform(-1.0, 1.0, size = [1, 4096])
    global seed
    global generator
    seed = seed+changes*0.02
    seed = (seed-np.min(seed))/(np.max(seed)-np.min(seed))*2-1
    image = generator.predict(changes)
    image = image[0, :, :, :]
    return np.uint8(image*255)
    
seed = np.random.uniform(-1.0, 1.0, size = [1, 4096])
generator_path = "g_out_410"
generator = tf.keras.models.load_model(generator_path)

while True:
    # Get a numpy array to display from the simulation
    npimage=getFrame()
    npimage = cv2.cvtColor(npimage, cv2.COLOR_BGR2RGB)
    cv2.imshow('image',npimage)
    cv2.waitKey(1)
    time.sleep(0.5)
