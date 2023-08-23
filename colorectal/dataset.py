import tensorflow as tf
import tensorflow_datasets as tfds
from colorectal import constants


def dataset_download():
    """
    The function downloads the colorectal dataset whit tfds->https://www.tensorflow.org/datasets/api_docs/python/tfds.
    :return: 	the colorectal dataset in tf.data.Dataset format, dataset info
    """
    [_ds], _ds_info = tfds.load(
        'colorectal_histology',
        split=['train'],
        as_supervised=True,
        shuffle_files=False,
        with_info=True)
    return _ds, _ds_info


def dataset_split(ds):
    """
    This function divides the dataset ds into three parts: train, validation, and test set. At first, the dataset is
    shuffled, then it is split into 8 different datasets, one for each class of texture. They are stored in a
    list _ds_category. So these datasets have 625 elements each (5000/8). For all i=0,...,7, _ds_category[i] is divided
    into three parts of 425, 100, 100 elements, named _ds_cat_train[i], _ds_cat_validation[i], and _ds_cat_test[i].
    Then to build ds_train (training set), every dataset in the _ds_category list is merged. The same idea applies to
    ds_validation and ds_test.
    This procedure is necessary to guarantee that in each set the proportion is the same as in the original dataset.

    :param ds: input dataset (colorectal dataset)
    :return: training set, validation set, test set
    """
    _ds = ds.shuffle(constants.DS_SIZE, reshuffle_each_iteration=False)
    _ds_category = []
    _ds_cat_train = []
    _ds_cat_validation = []
    _ds_cat_test = []
    _ds_cat_reference = []
    for i in range(8):
        _ds_category.append(ds.filter(lambda image, label: label == i))

    for i in range(8):
        _ds_cat_train.append(_ds_category[i].take(425))
        _ds_cat_validation.append(_ds_category[i].take(525).skip(425))
        _ds_cat_test.append(_ds_category[i].skip(525))

        _ds_cat_reference.append(_ds_cat_train[-1].take(4))

    _ds_train = _ds_cat_train[0]
    _ds_validation = _ds_cat_validation[0]
    _ds_test = _ds_cat_test[0]
    _ds_reference = _ds_cat_reference[0]
    for i in range(1, 8):
        _ds_train = _ds_train.concatenate(_ds_cat_train[i])
        _ds_validation = _ds_validation.concatenate(_ds_cat_validation[i])
        _ds_test = _ds_test.concatenate(_ds_cat_test[i])
        _ds_reference = _ds_reference.concatenate(_ds_cat_reference[i])
    return _ds_train, _ds_validation, _ds_test


def one_hot_encoding(dataset):
    """

    :param dataset: input dataset
    :return: the dataset taken in input where the category column is codified with the one hot encoding
    """
    def _one_h(image, label):
        return image, tf.one_hot(label, 8)
    return dataset.map(_one_h, num_parallel_calls=tf.data.AUTOTUNE)


def rotation(dataset):
    def _rot90(k: int):
        return lambda image, label: (tf.image.rot90(image, k=k), label)

    _dateset_rot1 = dataset.map(_rot90(1), num_parallel_calls=tf.data.AUTOTUNE)
    _dateset_rot2 = dataset.map(_rot90(2), num_parallel_calls=tf.data.AUTOTUNE)
    _dateset_rot3 = dataset.map(_rot90(3), num_parallel_calls=tf.data.AUTOTUNE)
    _dataset = dataset.concatenate(_dateset_rot1)
    _dataset = _dataset.concatenate(_dateset_rot2)
    _dataset = _dataset.concatenate(_dateset_rot3)
    return _dataset


def flip(dataset):
    def _flip_lr(image, label):
        return tf.image.flip_left_right(image), label

    return dataset.map(_flip_lr, num_parallel_calls=tf.data.AUTOTUNE)


def normalize(dataset):
    def _normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    return dataset.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)


def ca_sh_ba_pr(dataset):
    """
    This function improves performance of the dataset during training.
    https://www.tensorflow.org/guide/data_performance

    :param dataset: input dataset
    :return: the dataset cached, shuffled, divided in mini-batch and prefetched
    """
    _dataset = dataset.cache()
    _dataset = _dataset.shuffle(30000)
    _dataset = _dataset.batch(constants.BATCH_SIZE)
    _dataset = _dataset.prefetch(tf.data.AUTOTUNE)
    return _dataset
