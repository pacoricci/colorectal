import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from colorectal import constants
from sklearn.metrics import confusion_matrix


def loss_accuracy_graph(training_history, test_loss, test_cce_loss, test_ole_loss, test_accuracy):
    _n = len(training_history.history['loss'])
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    _axs[0].plot(range(_n), training_history.history['loss'], color='blue', label='Training loss')
    _axs[0].plot(range(_n),  training_history.history['val_loss'], color='orange', label='Validation loss')
    _axs[0].plot([_n-1],  [test_loss],'o', color='red', label='Test loss')
    _axs[0].legend()
    _axs[0].set_title('Loss: CCE loss + OLÉ loss')
    ################################
    _axs[1].plot(range(_n), training_history.history['cce_accuracy'], color='blue', label='Training accuracy')
    _axs[1].plot(range(_n),  training_history.history['val_cce_accuracy'], color='orange', label='Validation accuracy')
    _axs[1].plot([_n-1],  [test_accuracy],'o', color='red', label='Test accuracy')
    _axs[1].legend()
    _axs[1].set_title('Accuracy')
    ################################

    _fig2, _axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    _axs[0].plot(range(_n), training_history.history['cce_loss'], color='blue', label='Training CCE loss')
    _axs[0].plot(range(_n),  training_history.history['val_cce_loss'], color='orange', label='Validation CCE loss')
    _axs[0].plot([_n-1],  [test_cce_loss],'o', color='red', label='Test CCE loss')
    _axs[0].legend()
    _axs[0].set_title('CCE Loss')
    ################################
    _axs[1].plot(range(_n), training_history.history['ole_loss'], color='blue', label='Training OLÉ loss')
    _axs[1].plot(range(_n),  training_history.history['val_ole_loss'], color='orange', label='Validation OLÉ loss')
    _axs[1].plot([_n-1],  [test_ole_loss],'o', color='red', label='Test OLÉ loss')
    _axs[1].legend()
    _axs[1].set_title('OLÉ Loss')
    ################################
    _fig.savefig('graph_loss_acc.pdf', format='pdf')
    _fig2.savefig('graph_cce_ole_loss.pdf', format='pdf')


def confusion_matrix_graph(model, ds_test):
    _predict_y = model.predict(ds_test)[0]
    _classes_y = np.argmax(_predict_y, axis=1)
    _cm = confusion_matrix(ds_test['label'], _classes_y)
    _df_cm = pd.DataFrame(_cm, constants.NAMES, constants.NAMES)
    fig = plt.figure(figsize=(9, 7), tight_layout=True)
    sn.set(font_scale=0.8)  # for label size
    sn.heatmap(_df_cm, annot=True, annot_kws={"size": 16}, fmt='g')  # font size
    fig.savefig('confusion_matrix.pdf', format='pdf')
