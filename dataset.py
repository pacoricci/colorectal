import tensorflow as tf
import tensorflow_datasets as tfds
import constants


def dataset_download():
    [_ds], _ds_info = tfds.load(
        'colorectal_histology',
        split=['train'],
        as_supervised=True,
        shuffle_files=False,
        with_info=True)
    return _ds, _ds_info


def dataset_split(ds):
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
    _dataset = dataset.cache()
    _dataset = _dataset.shuffle(30000)
    _dataset = _dataset.batch(constants.BATCH_SIZE)
    _dataset = _dataset.prefetch(tf.data.AUTOTUNE)
    return _dataset
