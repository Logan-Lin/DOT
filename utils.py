import os
from datetime import datetime

import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from model.trainer import cal_regression_metric


time_format = "t%Y_%m_%d_%H_%M_%S"


def list_all_results(name):
    results = []
    filepath = os.path.join('result', 'result', name + '.h5')
    if not os.path.exists(filepath):
        raise FileNotFoundError('File not existed.')

    with h5py.File(filepath) as h5file:
        ts = list(h5file.keys())
    ts = map(lambda x: datetime.strptime(x, time_format), ts)
    ts = pd.Series(ts).sort_values()
    for t in ts:
        result = pd.read_hdf(filepath, key=datetime.strftime(t, time_format))
        result['name'] = name
        result['t'] = t
        results.append(result)
    results = pd.DataFrame(results)
    results = results.set_index(['name', 't'])

    return results


def list_multi_results(files):
    results = []
    for file in files:
        result = get_lastest_result(file)
        results.append(result)
    results = pd.DataFrame(results)
    return results


def get_lastest_result(name):
    all_results = list_all_results(name)
    return all_results.iloc[-1]


def cal_regression_result(name, label_p=0.0, noise_p=0.0):
    np.random.seed(1000)

    loaded_pre = np.load(os.path.join('result', 'prediction', name + '.npz'))
    pre, label = loaded_pre['pre'], loaded_pre['label']
    size = pre.shape[0]

    label_s = int(size * label_p)
    pre[:label_s] = label[:label_s]

    noise_s = int(size * noise_p)
    np.random.shuffle(pre[:noise_s])

    rmse, mae, mape = cal_regression_metric(label, pre, p=False)
    print(f'{name}: rmse %.3f, mae %.3f, mape %.3f' % (rmse, mae, mape * 100))
    return rmse, mae, mape


def cal_inference_result(name):
    loaded = np.load(os.path.join('result', 'prediction', name + '.npz'))
    pre, label = loaded['pre'], loaded['label']
    print('Overall:')
    cal_regression_metric(label, pre)

    pre = pre.reshape(pre.shape[0], 3, -1)
    label = label.reshape(label.shape[0], 3, -1)
    for c in range(3):
        print(f'Channel : {c+1}')
        cal_regression_metric(label[:, c], pre[:, c])


def plot_single_axis(x, d, lim, title=None, name='undefined', ylabel=None):
    figure = plt.figure(figsize=(3, 1.7), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(d, marker='o', color='black')
    ax.set_xticks(range(len(d)), x)
    ax.set_ylim(lim)
    ax.tick_params(axis='both', direction='in')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join('result', 'plot', name + '.pdf'), format='pdf',
                bbox_inches='tight', transparent=True)
    plt.close()


def plot_dual_single_axis(x, d1, d2, lim, title=None, name='undefined', ylabel=None, legend=None):
    figure = plt.figure(figsize=(3, 1.7), dpi=150)
    ax = figure.add_subplot(111)
    ax.plot(d1, marker='o', color='#1A237E')
    ax.plot(d2, marker='v', color='#BF360C')
    ax.set_xticks(range(len(d1)), x)
    if legend is not None:
        ax.legend(legend, ncols=1)
    ax.set_ylim(lim)
    ax.tick_params(axis='both', direction='in')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join('result', 'plot', name + '.pdf'), format='pdf',
                bbox_inches='tight', transparent=True)
    plt.close()


def plot_double_axis(x, d1, d2, title, lim1=None, lim2=None, name='undefined',
                     ylabel=['RMSE (minute)', 'MAE (minute)']):
    figure = plt.figure(figsize=(3, 1.7), dpi=150)
    c1 = '#1A237E'
    c2 = '#BF360C'

    if lim1 is None:
        lim1 = [min(d1) - 1, max(d1) + 0.5]
    if lim2 is None:
        lim2 = [min(d2) - 0.5, max(d2) + 1]

    ax1 = figure.add_subplot(111)
    p1 = ax1.plot(d1, color=c1, marker='o')
    ax1.set_xticks(range(len(d1)), x)
    ax1.set_ylim(lim1)
    ax1.tick_params(axis='both', direction='in')
    ax1.tick_params(axis='y', colors=c1)
    ax1.set_ylabel(ylabel[0], color=c1)

    ax2 = ax1.twinx()
    p2 = ax2.plot(d2, color=c2, marker='v')
    ax2.tick_params(axis='y', colors=c2, direction='in')
    ax2.set_ylim(lim2)
    ax2.set_ylabel(ylabel[1], color=c2)

    # plt.legend(handles=p1+p2, labels=['RMSE', 'MAE'], ncol=2, loc='best')
    plt.title(title)

    plt.savefig(os.path.join('result', 'plot', name + '.pdf'), format='pdf',
                bbox_inches='tight')
    plt.close()


def plot_image_compare(img1, img2, name='undefined'):
    cmaps = ['GnBu', 'YlGn', 'OrRd']
    num_c = img1.shape[0]
    figure = plt.figure(figsize=(num_c*4, 2*4), dpi=150)

    for c in range(num_c):
        ax1 = figure.add_subplot(2, num_c, c + 1)
        ax1.invert_yaxis()
        ax1.imshow(img1[c], cmap=cmaps[c % 3])
        ax1.axis('off')

        ax2 = figure.add_subplot(2, num_c, num_c + c + 1)
        ax2.invert_yaxis()
        ax2.imshow(img2[c], cmap=cmaps[c % 3])
        ax2.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join('result', 'plot', name + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def plot_bar(d, name='undefined'):
    colors = ['#9CCC65', '#02fdff', '#3cc3ff', '#807fff', '#c03fff', '#fc02ff', '#ff7323']

    bar_width = 0.1
    fig, ax = plt.subplots(figsize=(3, 2), dpi=150)
    for i, v in enumerate(d):
        ax.bar(bar_width * i, v, bar_width, color=colors[i])
    plt.xticks([], [])

    min_max = [np.min(d), np.max(d)]
    span = min_max[1] - min_max[0]
    plt.ylim([min_max[0] - span * 0.1, min_max[1] + span * 0.1])
    plt.savefig(os.path.join('result', 'plot', f'{name}.pdf'), format='pdf',
                bbox_inches='tight', transparent=True)
    plt.close()


def prob_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size
