#!/usr/bin/env python2
#coding: utf-8
from __future__ import division
import os
import shutil
import csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')
import seaborn as sns
from pandas.tseries.offsets import Milli
import datetime
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import optunity
import optunity.metrics
from optunity.constraints import wrap_constraints
from optunity.solvers.GridSearch import GridSearch
from optunity.solvers.ParticleSwarm import ParticleSwarm
import multiprocessing
import gc

try:
    import cPickle as pickle
except ImportError:
    import pickle


def pp(pic_path, accuracy_list, sigsum, real_sigsum, adjusted_sigsum):
    plt.figure()
    plt.subplot(211)
    plt.plot(accuracy_list, label='$accuracy$')
    plt.legend()
    plt.ylabel('accuracy')
    # plt.figtext(0.39, 0.95, 'accuracy:{:.4f}'.format(accuracy_list[-1]), color='green')
    # plt.figtext(0.13, 0.91, 'sigsum:{:6.1f}'.format(sigsum[-1]), color='green')
    # plt.figtext(0.39, 0.91, 'real_sigsum:{:6.1f}'.format(real_sigsum[-1]), color='green')
    # plt.figtext(0.65, 0.91, 'adjusted_sigsum:{:6.1f}'.format(adjusted_sigsum[-1]), color='green')
    plt.title('accuracy:{:.4f}, sigsum:{:6.1f}, real_sigsum:{:6.1f}, adjusted_sigsum:{:6.1f}'.format(
        accuracy_list[-1], sigsum[-1], real_sigsum[-1], adjusted_sigsum[-1]))
    plt.subplot(212)
    plt.plot(sigsum, 'r-', label='$sigsum$')
    plt.plot(real_sigsum, 'g-', label='$realsigsum$')
    plt.plot(adjusted_sigsum, 'b-', label='$adjustedsigsum$')
    plt.legend()
    plt.ylabel('sigsum')
    plt.savefig(pic_path)
    plt.close()
    plt.figure()
    plt.plot(adjusted_sigsum, 'b-', label='$adjustedsigsum$')
    plt.legend()
    plt.ylabel('adjusted_sigsum')
    # plt.figtext(0.39, 0.95, 'adjusted_sigsum:{:6.1f}'.format(adjusted_sigsum[-1]), color='green')
    plt.title('adjusted_sigsum:{:6.1f}'.format(adjusted_sigsum[-1]))
    plt.savefig(os.path.splitext(pic_path)[0]+'_'+os.path.splitext(pic_path)[1])
    plt.close()

if __name__ == '__main__':

    ##########################################################################################
    '''
    show the results
    '''
    plt.figure()

    results_path = './results_dir_online/results_resample1timeshift1/results.pkl'
    with open(results_path, 'rb') as fp:
        (accuracy_list, sigsum, real_sigsum, adjusted_sigsum) = pickle.load(fp)
    print 'accuracy: %.4f, sigsum: %6.1f, real_sigsum: %6.1f, adjusted_sigsum: %6.1f' % \
           (accuracy_list[-1], sigsum[-1], real_sigsum[-1], adjusted_sigsum[-1])
    # pp('./accuracy_sigsum.png', accuracy_list, sigsum, real_sigsum, adjusted_sigsum)
    plt.plot(adjusted_sigsum, label='$online$')

    results_path = './results_dir_offline/results_resample1timeshift1/results.pkl'
    with open(results_path, 'rb') as fp:
        (accuracy_list, sigsum, real_sigsum, adjusted_sigsum) = pickle.load(fp)
    print 'accuracy: %.4f, sigsum: %6.1f, real_sigsum: %6.1f, adjusted_sigsum: %6.1f' % \
           (accuracy_list[-1], sigsum[-1], real_sigsum[-1], adjusted_sigsum[-1])
    # pp('./accuracy_sigsum.png', accuracy_list, sigsum, real_sigsum, adjusted_sigsum)
    plt.plot(adjusted_sigsum, label='$offline$')

    plt.legend()
    plt.ylabel('adjusted_sigsum')
    # plt.figtext(0.39, 0.95, 'adjusted_sigsum:{:6.1f}'.format(adjusted_sigsum[-1]), color='green')
    # plt.title('adjusted_sigsum:{:6.1f}'.format(adjusted_sigsum[-1]))
    plt.savefig('./adjusted_sigsum.png')
