import tensorflow as tf

model_dir="/Users/alex/funstuff-local/gan-test/gan-tests/display/g_out_410/"

converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
tflite_model = converter.convert()

with open('../generator.tflite', 'wb') as f:
  f.write(tflite_model)
