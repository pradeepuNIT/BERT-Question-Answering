import tensorflow as tf
import os

#path that contains all 3 ckpt files of your fine-tuned model
path = './models/SQUAD_2_COPY'

#path to output the new optimized model
output_path = os.path.join(path, 'models/optimized_squad2_model')

sess = tf.Session()
imported_meta = tf.train.import_meta_graph(os.path.join(path, 'model.ckpt-43439.meta'))
imported_meta.restore(sess, os.path.join(path, 'model.ckpt-43439'))
my_vars = []
for var in tf.all_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
        my_vars.append(var)
saver = tf.train.Saver(my_vars)
saver.save(sess, os.path.join(output_path, 'model.ckpt'))