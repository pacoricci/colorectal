import tensorflow as tf

import colorectal as cl

tf.random.set_seed(cl.constants.TF_SEED)
# Dataset download
ds, ds_info = cl.dataset.dataset_download()
# Normalization of images: image_tensor = image_tensor/255
ds = cl.dataset.normalize(ds)
# Dataset division in training set, validation set, test set
ds_train, ds_validation, ds_test = cl.dataset.dataset_split(ds)
# One Hot Encoding
ds_train = cl.dataset.one_hot_encoding(ds_train)
ds_validation = cl.dataset.one_hot_encoding(ds_validation)
ds_test = cl.dataset.one_hot_encoding(ds_test)
# Dataset augmentation
ds_train = cl.dataset.rotation(ds_train)
ds_train = cl.dataset.flip(ds_train)
# Cache, shuffle, batch, prefetch
ds_train = cl.dataset.ca_sh_ba_pr(ds_train)
ds_validation = cl.dataset.ca_sh_ba_pr(ds_validation)
ds_test = cl.dataset.ca_sh_ba_pr(ds_test)
#
#
# Generate the model
model = cl.cnn_model.gen_model()
# Model training
history = cl.training.fit(model, ds_train, ds_validation)
# Evaluate the model on test set
test_loss, test_cce_loss, test_ole_loss, test_accuracy = cl.training.evaluate(model, ds_test)
cl.graphs.loss_accuracy_graph(history, test_loss, test_cce_loss, test_ole_loss, test_accuracy)
cl.graphs.confusion_matrix_graph(model, ds_test)




