import tensorflow as tf
import numpy as np
import cv2
    
seed = np.random.uniform(-1.0, 1.0, size = [1, 4096])
generator_path = "g_out_410"
generator = tf.keras.models.load_model(generator_path)

def getFrame():
    global seed
    changes = np.random.uniform(-1.0, 1.0, size = [1, 4096])
    seed = seed+changes*0.02
    seed = (seed-np.min(seed))/(np.max(seed)-np.min(seed))*2-1
    image = generator.predict(seed)
    image = image[0, :, :, :]
    return np.uint8(image*255)

if __name__=="__main__":
    while True:
        # Get a numpy array to display from the simulation
        npimage=getFrame()
        npimage = cv2.cvtColor(npimage, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',npimage)
        cv2.waitKey(1)
