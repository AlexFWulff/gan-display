import tensorflow as tf
import numpy as np

model_path = "../generator.tflite"

interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
#input_data = np.array(np.random.uniform(-1.0, 1.0, size = [1, 4096]), dtype=np.float32)

#interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)



