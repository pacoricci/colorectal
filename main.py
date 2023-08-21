import tensorflow as tf

import constants
import dataset
import cnn_model
import training

tf.random.set_seed(constants.TF_SEED)
# Dataset download
ds, ds_info = dataset.dataset_download()
# Normalization of images: image_tensor = image_tensor/255
ds = dataset.normalize(ds)
# Dataset division in training set, validation set, test set
ds_train, ds_validation, ds_test = dataset.dataset_split(ds)
# One Hot Encoding
ds_train = dataset.one_hot_encoding(ds_train)
ds_validation = dataset.one_hot_encoding(ds_validation)
ds_test = dataset.one_hot_encoding(ds_test)
# Dataset augmentation
ds_train = dataset.rotation(ds_train)
ds_train = dataset.flip(ds_train)
# Cache, shuffle, batch, prefetch
ds_train = dataset.ca_sh_ba_pr(ds_train)
ds_validation = dataset.ca_sh_ba_pr(ds_validation)
ds_test = dataset.ca_sh_ba_pr(ds_test)
#
#
# Generate the model
model = cnn_model.gen_model()
# Model training
history = training.fit(model, ds_train, ds_validation)
# Evaluate the model on test set
test_loss, test_cce_loss, test_ole_loss, test_accuracy = training.evaluate(model, ds_test)




