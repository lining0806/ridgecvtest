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
    show parameters distribution
    '''
    # best_parameters_path = './best_parameters'
    # with open(best_parameters_path, 'rb') as fp:
    #     (best_params, optimum, df_sort) = pickle.load(fp)
    # print 'best parameters:', best_params
    # print 'best score:', optimum
    # print 'sorted best parameters:'
    # print df_sort
    #
    # # df_sort.plot.scatter(x='maxlag', y='threshold')
    # # plt.show()
    # # df_sort.plot.scatter(x='threshold', y='windowsize')
    # # plt.show()
    # # df_sort.plot.scatter(x='windowsize', y='maxlag')
    # # plt.show()
    # # df_sort.plot.scatter(x='maxlag', y='threshold', c='windowsize', s=50)
    # # plt.show()
    #
    # ####
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # for xs, ys, zs in zip(x, y, z):
    # #     ax.scatter(xs, ys, zs)
    # ax.scatter(df_sort['maxlag'], df_sort['windowsize'], df_sort['threshold'])
    # ax.set_xlabel('maxlag')
    # ax.set_ylabel('windowsize')
    # ax.set_zlabel('threshold')
    # # plt.show()
    # # ####
    # # sns.set()
    # # sns.pairplot(df_sort.drop('value', axis=1))
    # # # plt.show()
    # plt.savefig('./best_parameters.png')

    ##########################################################################################
    '''
    show the results
    '''
    results_path = './results.pkl'
    with open(results_path, 'rb') as fp:
        (accuracy_list, sigsum, real_sigsum, adjusted_sigsum) = pickle.load(fp)
    print 'accuracy: %.4f, sigsum: %6.1f, real_sigsum: %6.1f, adjusted_sigsum: %6.1f' % \
           (accuracy_list[-1], sigsum[-1], real_sigsum[-1], adjusted_sigsum[-1])
    pp('./accuracy_sigsum.png', accuracy_list, sigsum, real_sigsum, adjusted_sigsum)

    ####
    plt.figure()
    plt.plot(adjusted_sigsum, label='$adjustedsigsum$')
    plt.legend()
    plt.ylabel('adjusted_sigsum')
    # plt.figtext(0.39, 0.95, 'adjusted_sigsum:{:6.1f}'.format(adjusted_sigsum[-1]), color='green')
    plt.title('adjusted_sigsum:{:6.1f}'.format(adjusted_sigsum[-1]))
    plt.savefig('./adjusted_sigsum.png')

    ##########################################################################################
    '''
    show the results for different resample and timeshift
    '''
    # # resample_num_list = [1]
    # # resample_num_list = [1, 2, 4, 8]
    # resample_num_list = [1, 2, 3, 4, 5, 6, 7, 8]
    # # timeshift_num_list = [1]
    # # timeshift_num_list = [1, 2, 4, 8, 16]
    # timeshift_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]
    #
    # # results_dir = './'
    # results_dir = './results_dir/'
    # results_list = []
    #
    # for resample_num in resample_num_list:
    #
    #     '''you can change the condition here: whether timeshift equals resample or not'''
    #     # timeshift_num_list = [resample_num] ## timeshift == resample
    #     timeshift_num_list = filter(lambda x: x>=resample_num, timeshift_num_list) ## timeshift >= resample
    #
    #     plt.figure(0)
    #     plt.figure(1)
    #     plt.figure(2)
    #     plt.figure(3)
    #
    #     for timeshift_num in timeshift_num_list:
    #         results_path = results_dir+'results_'+'resample'+str(resample_num)+'timeshift'+str(timeshift_num)
    #         with open(os.path.join(results_path, 'results.pkl'), 'rb') as fp:
    #             (accuracy_list, sigsum, real_sigsum, adjusted_sigsum) = pickle.load(fp)
    #         print 'accuracy: %.4f, sigsum: %6.1f, real_sigsum: %6.1f, adjusted_sigsum: %6.1f' % \
    #                (accuracy_list[-1], sigsum[-1], real_sigsum[-1], adjusted_sigsum[-1])
    #         # pp(os.path.join(results_path, './accuracy_sigsum.png'), accuracy_list, sigsum, real_sigsum, adjusted_sigsum)
    #         filter_results = filter(lambda x:x.startswith('accuracy_sigsum_tradecount'), os.listdir(results_path))
    #         tradecount = int(filter_results[0].replace('accuracy_sigsum_tradecount', '').replace('.png', ''))
    #
    #         results_list.append(
    #             {
    #                 'resample':resample_num,
    #                 'timeshift':timeshift_num,
    #                 'accuracy':accuracy_list[-1],
    #                 'sigsum':sigsum[-1],
    #                 'realsigsum':real_sigsum[-1],
    #                 'adjustedsigsum':adjusted_sigsum[-1],
    #                 'tradecount':tradecount,
    #                 'sigpertrade':adjusted_sigsum[-1]/tradecount,
    #             }
    #         )
    #
    #         plt.figure(0)
    #         plt.title('accuracy')
    #         plt.plot(accuracy_list, label='$timeshift%d$' % timeshift_num)
    #         plt.figure(1)
    #         plt.title('sigsum')
    #         plt.plot(sigsum, label='$timeshift%d$' % timeshift_num)
    #         plt.figure(2)
    #         plt.title('realsigsum')
    #         plt.plot(real_sigsum, label='$timeshift%d$' % timeshift_num)
    #         plt.figure(3)
    #         plt.title('adjustedsigsum')
    #         plt.plot(adjusted_sigsum, label='$timeshift%d$' % timeshift_num)
    #
    #     plt.figure(0)
    #     plt.legend()
    #     plt.savefig(results_dir+'accuracy_resample%d' % resample_num)
    #     plt.close()
    #     plt.figure(1)
    #     plt.legend()
    #     plt.savefig(results_dir+'sigsum_resample%d' % resample_num)
    #     plt.close()
    #     plt.figure(2)
    #     plt.legend()
    #     plt.savefig(results_dir+'realsigsum_resample%d' % resample_num)
    #     plt.close()
    #     plt.figure(3)
    #     plt.legend()
    #     plt.savefig(results_dir+'adjustedsigsum_resample%d' % resample_num)
    #     plt.close()
    #
    # df_results = pd.DataFrame(results_list)
    # print df_results
    #
    # with open(results_dir+'df_results.pkl', 'wb') as fp:
    #     pickle.dump(df_results, fp)
    #
    # with open(results_dir+'df_results.pkl', 'rb') as fp:
    #     df_results = pickle.load(fp)
    #
    # print df_results.sort_values(by='adjustedsigsum', axis=0, ascending=False).head(20)
    # plt.figure()
    # df_results.plot.scatter(x='resample', y='timeshift', c='accuracy', s=50)
    # plt.savefig(results_dir+'accuracylist_resample_timeshift.png')
    # plt.close()
    # plt.figure()
    # df_results.plot.scatter(x='resample', y='timeshift', c='sigsum', s=50)
    # plt.savefig(results_dir+'sigsum_resample_timeshift.png')
    # plt.close()
    # plt.figure()
    # df_results.plot.scatter(x='resample', y='timeshift', c='realsigsum', s=50)
    # plt.savefig(results_dir+'realsigsum_resample_timeshift.png')
    # plt.close()
    # plt.figure()
    # df_results.plot.scatter(x='resample', y='timeshift', c='adjustedsigsum', s=50)
    # plt.savefig(results_dir+'adjustedsigsum_resample_timeshift.png')
    # df_results.plot.scatter(x='resample', y='timeshift', s=df_results['adjustedsigsum']*0.1)
    # plt.savefig(results_dir+'adjustedsigsum_resample_timeshift_.png')
    #
    # df_results.sort_values(by='adjustedsigsum', axis=0, ascending=False, inplace=True)
    # df_results.to_excel(results_dir+'output.xlsx', 'df_results')
