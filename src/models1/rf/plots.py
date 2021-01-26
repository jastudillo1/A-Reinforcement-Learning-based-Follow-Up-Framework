import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
        
def plot_cm_ax(ax, cm, classes, normalize, title):
    
    ax.set_title(title)

    labels = classes#[self.trans[i] for i in range(self.num_classes)]
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm[i,j] ='%.2f' %cm[i,j]

    thresh = 0.001
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] < thresh else 'black')

    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


def plot_cm(y, y_pred, classes, descr, descr_save, plot_dir):

    plt.clf()
    plt.rc('font', size=15)
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=15)
    plt.rc('figure', titlesize=22)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20,12))
    cm = confusion_matrix(y, y_pred)
    title = 'Confusion Matrix ' + descr + ' features'
    plot_cm_ax(ax0, cm, normalize=False, classes=classes, title='Normalized '+ title)
    plot_cm_ax(ax1, cm, normalize=True, classes=classes, title=title)

    plt.tight_layout()
    fig.savefig(f'{plot_dir}/{descr_save}.png', dpi=120)