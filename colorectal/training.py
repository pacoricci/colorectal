import tensorflow as tf
from colorectal import constants
from colorectal.losses import ole_loss
from keras.optimizers import SGD


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_cce_loss",
    patience=15,
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=80,
)


def fit(model, ds_train, ds_validation):
    model.compile(optimizer=SGD(constants.SGD_LEARNING_RATE, momentum=constants.MOMENTUM),
                  loss=[tf.keras.losses.CategoricalCrossentropy(), ole_loss],
                  loss_weights=[1., constants.OLE_WEIGHT],
                  metrics={'cce': 'accuracy'})
    _history = model.fit(ds_train,
                         epochs=constants.EPOCHS,
                         validation_data=ds_validation,
                         verbose=1,
                         callbacks=[early_stopping])
    return _history


def evaluate(model, ds_test):
    model.compile(loss=[tf.keras.losses.CategoricalCrossentropy(), ole_loss],
                  loss_weights=[1.0, constants.OLE_WEIGHT],
                  metrics={'cce': 'accuracy'})
    _test_loss, _test_cce_loss, _test_ole_loss, _test_accuracy = model.evaluate(ds_test)
    return _test_loss, _test_cce_loss, _test_ole_loss, _test_accuracy
