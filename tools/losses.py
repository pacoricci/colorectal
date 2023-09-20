import tensorflow as tf


def ole_loss(y_true, y_pred):
    """
    This function defines the OLE loss (Orthogonal Low-rank Embedding) in TensorFlow.
    :param y_true: mini-batch of targets
    :param y_pred: mini-batch of model predictions
    :return: OLE loss
    """
    _y_true = tf.cast(y_true,tf.int32)
    _y_true = tf.cast(_y_true,tf.bool)
    _y_true = tf.transpose(_y_true)
    _y_true = tf.expand_dims(_y_true, -1)
    _mult = tf.where(_y_true, y_pred, tf.zeros_like(y_pred))
    _s = tf.expand_dims(y_pred, 0)
    _mult = tf.concat([_s, _mult], axis= 0)
    _svd = tf.linalg.svd(_mult, compute_uv=False)
    _nns = tf.reduce_sum(_svd, axis=1)
    _max = tf.math.maximum(tf.constant([1.]), _nns[1:])
    return tf.reduce_sum(_max) - _nns[0]
