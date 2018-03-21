#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
__author__ = 'LiNing'


from sigknn import *
from sigsvm import *

import os, sys, datetime, glob
import argparse, logging
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import pylab as pl
import matplotlib.pyplot as plt
import cPickle as pickle
from pprint import pprint
import multiprocessing as mp

import csv
import talib
from talib import abstract, common, func

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2, f_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score

import optunity, optunity.metrics
from optunity.constraints import wrap_constraints

from zigzag import peak_valley_pivots, max_drawdown


## set globle variables ##
APP_DIR = os.path.split(__file__)[0]
sys.path.append(os.path.join(APP_DIR, 'lib'))
APP_NAME = os.path.split(os.path.splitext(__file__)[0])[1]

## set logger ##
def init_log(logger_name=''):
    ## get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ## A handler class which writes formatted logging records to disk files.
    filehandler = logging.FileHandler('%s.log' % (datetime.datetime.now().strftime('%Y-%m-%d')), mode='w')
    # filehandler = logging.FileHandler('log.txt', mode='w')
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    ## A handler class which writes formatted logging records to disk files.
    consolehandler = logging.StreamHandler(sys.stdout)
    consolehandler.setFormatter(formatter)
    logger.addHandler(consolehandler)

    return logger

## app initiation here
logger = init_log(APP_NAME)


def isDayAcross(timestamp, diff_n=1):
    trading_time = timestamp.time()
    close_time = datetime.time(15, 15-diff_n)
    if trading_time > close_time or trading_time == datetime.time(11, 30):
        return True
    else:
        return False

def isInTradingTime(timestamp):
    trading_time = timestamp.time()
    starttime1 = datetime.time(9, 15)
    stoptime1 = datetime.time(11, 30)
    starttime2 = datetime.time(13, 0)
    stoptime2 = datetime.time(15, 15)
    if starttime1 <= trading_time <= stoptime1 or starttime2 <= trading_time <= stoptime2:
        return True
    else:
        return False

def setTimeResample(rs, delta = '5min'):
    newopen = rs.open.resample(delta, how = 'first')
    newhigh = rs.high.resample(delta, how = 'max')
    newlow = rs.low.resample(delta, how = 'min')
    newclose = rs.close.resample(delta, how = 'last')
    newvolume = rs.volume.resample(delta, how = 'sum')
    newrs = pd.DataFrame({'open':newopen, 'high':newhigh, 'low':newlow, 'close':newclose, 'volume':newvolume})
    return newrs.dropna()

def setNumDeltaReverse(rs, delta = 5):
    ## from end to towards begin to resample
    rs = rs.sort_index(ascending = False)

    index_array = np.arange(0, rs.shape[0]-delta+1, delta)
    # print index_array
    newopen = [rs.open.ix[i+delta-1] for i in index_array] # the last
    newhigh = [max(rs.high.ix[i:i+delta]) for i in index_array]
    newlow = [min(rs.low.ix[i:i+delta]) for i in index_array]
    newclose = [rs.close.ix[i] for i in index_array] # the first
    newvolume = [sum(rs.volume.ix[i:i+delta]) for i in index_array]
    newrs = pd.DataFrame({'open':newopen, 'high':newhigh, 'low':newlow, 'close':newclose, 'volume':newvolume})
    newrs.index = rs.index[[i for i in index_array]]

    newrs = newrs.sort_index(ascending = True)
    return newrs.dropna()

def loadTimeSeries(csv_path, rs_num = 10000, nrows = 10000, resample_time = 5):

    lines = sum(1 for _ in csv.reader(open(csv_path)))
    ## ------------------------------------------------------------------------------------
    assert rs_num >= nrows
    rs = pd.read_csv(csv_path, header = None, skiprows = lines-rs_num, nrows = nrows)
    # rs = pd.read_csv(csv_path, header = None, skiprows = lines-rs_num, nrows = 10000)
    # # rs = pd.read_csv(csv_path, skiprows = lines-rs_num, nrows = 9999)
    ## ------------------------------------------------------------------------------------
    # print rs
    rs = rs.ix[:, 0:6] ## 0,1,2,3,4,5,6 is the column index, not number
    rs.columns = ['Date', 'Time', 'open', 'high', 'low', 'close', 'volume']
    rs.index = pd.to_datetime(rs.Date+' '+rs.Time)
    ######################################################################
    '''remove the same timestamp'''
    position_dict = {}
    k, last_index = 0, None
    for index in rs.index:
        if index != last_index:
            position_dict[index] = k
        k += 1
        last_index = index
    rs = rs.ix[sorted(position_dict.values())]
    ######################################################################
    # index = [item for item in rs.index if isInTradingTime(item)]
    # rs = rs.ix[index]
    del rs['Date']
    del rs['Time']
    # print rs
    ######################################################################
    # data = rs ## no resample training data for forwardtest when skipping by resample step
    if isinstance(resample_time, str):
        data = setTimeResample(rs, delta = resample_time)
    elif isinstance(resample_time, int):
        data = setNumDeltaReverse(rs, delta = resample_time)
    else:
        data = None
    # print data
    ######################################################################

    return data

def target_define(data, threshold = 1):
    data['pos'] = (data['diff']>threshold).astype(int)
    data['neg'] = -(data['diff']<-threshold).astype(int)
    data['label'] = data['pos']+data['neg']
    return data

def threshold_define(data, zero_propotion = 0.1):
    #######################################################################################
    diff_values = sorted(np.abs(data['diff']))
    threshold = diff_values[int(len(diff_values)*zero_propotion)]
    #######################################################################################
    # print 'threshold:', threshold
    return threshold

def mean_classification(dataset_window, test_size):
    diff_pred = []
    for i in np.arange(test_size, 0 , -1):
        X_train = dataset_window.shift(i-1).dropna()
        X_test = dataset_window.ix[-i]
        mean_train = np.mean(X_train)
        diff_test = mean_train-X_test
        diff_pred.append(diff_test)
    y_pred = np.sign(diff_pred)
    #######################################################################################
    return y_pred

def max_drawdown_num(X):
    '''
    Return the absolute value of the maximum drawdown of sequence X.

    Note
    ----
    If the sequence is strictly increasing, 0 is returned.
    '''
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        # dd = (peak - x) / peak
        dd = peak - x
        if dd > mdd:
            mdd = dd
    return mdd

def max_drawdown_rate(X):
    '''
    Return the absolute value of the maximum drawdown of sequence X.

    Note
    ----
    If the sequence is strictly increasing, 0 is returned.
    '''
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        # dd = peak - x
        if dd > mdd:
            mdd = dd
    return mdd

def resultplot_backup(op):
    rs = pd.read_csv(op.fn_out, sep='\t')
    '''here you must calculate diff using the next one close and current close!!!'''
    rs['diff'] = rs.close.diff(periods=1).shift(-1).fillna(0)
    # print rs
    ## ------------------------------------------------------------------------------------
    predict = np.array(rs['predict'])
    predict_bool = (predict!=0).astype(int)
    # print 'predict times:', sum(predict_bool)
    diff = np.array(rs['diff'])
    siglist = predict*diff
    correctlist = predict_bool & (siglist>=0).astype(int)
    ## plot
    plt.figure()
    plt.subplot(311)
    sigsumlist = np.cumsum(siglist)
    # print 'the last sigsum:', sigsumlist[-1]
    ## ------------------------------------------------------------------------------------
    # i = 0
    # while sigsumlist[i] == 0:
    #     sigsumlist[i] = 0.001
    #     i += 1
    # up_down_thresh = 10
    # pivots = peak_valley_pivots(sigsumlist, up_down_thresh, -up_down_thresh)
    # plt.plot(sigsumlist[pivots!=0], 'k:', label='$pivots$')
    ## ------------------------------------------------------------------------------------
    plt.plot(sigsumlist, 'k-', label='$sigsum$')
    plt.legend()
    plt.ylabel('sigsum')
    plt.ylim(-500, 3000)
    plt.subplot(312)
    averagesiglist = np.cumsum(siglist)/np.cumsum(predict_bool)
    # print 'the last averagesig:', averagesiglist[-1]
    plt.plot(averagesiglist, label='$averagesig$')
    plt.legend()
    plt.ylabel('averagesig')
    plt.ylim(0, 1)
    plt.subplot(313)
    accuracylist = np.cumsum(correctlist)/np.cumsum(predict_bool)
    # print 'the last accuracy:', accuracylist[-1]
    plt.plot(accuracylist, label='$accuracy$')
    plt.legend()
    plt.ylabel('accuracy')
    plt.ylim(0.4, 0.8)
    plt.figtext(0.15, 0.95, 'predict times:{}'.format(sum(predict_bool)), color='green')
    plt.figtext(0.4, 0.95, 'mdd num:{:.2f}'.format(max_drawdown_num(sigsumlist)), color='green')
    # plt.figtext(0.7, 0.95, 'mdd rate:{:.2f}'.format(max_drawdown_rate(sigsumlist[sigsumlist>200])), color='green')
    plt.figtext(0.15, 0.91, 'last sigsum:{:.1f}'.format(sigsumlist[-1]), color='green')
    plt.figtext(0.4, 0.91, 'last averagesig:{:.3f}'.format(averagesiglist[-1]), color='green')
    plt.figtext(0.7, 0.91, 'last accuracy:{:.2f}'.format(accuracylist[-1]), color='green')
    fig_dir = os.path.split(op.pn_out)[0]
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(op.pn_out)
    plt.close()

def resultplot(op):
    #######################################################################################
    # rs = pd.read_csv(op.fn_out, index_col=0, sep='\t')
    rs = pd.read_csv(op.fn_out, sep='\t')
    rs.index = pd.to_datetime(rs['index'])
    del rs['index']
    # print rs
    ## ------------------------------------------------------------------------------------
    '''here you must calculate diff using the next one close and current close!!!'''
    rs['diff'] = rs.close.diff(periods=1).shift(-1).fillna(0)
    rs['sig'] = rs['predict']*rs['diff']
    rs['sigsum'] = np.cumsum(rs['sig'])
    # print rs
    ## ------------------------------------------------------------------------------------
    rs['predict_bool'] = (rs['predict']!=0).astype(int)
    rs['correct'] = rs['predict_bool'] & (rs['sig']>=0).astype(int)
    # print rs
    ## ------------------------------------------------------------------------------------
    ## plot
    plt.figure()
    plt.subplot(311)
    # print 'the last sigsum:', rs['sigsum'][-1]
    plt.plot(rs['sigsum'], 'k-', label='$sigsum$')
    plt.legend()
    plt.ylabel('sigsum')
    plt.ylim(-500, 3000)
    plt.subplot(312)
    rs['averagesig'] = rs['sigsum']/np.cumsum(rs['predict_bool'])
    # print 'the last averagesig:', rs['averagesig'][-1]
    plt.plot(rs['averagesig'], label='$averagesig$')
    plt.legend()
    plt.ylabel('averagesig')
    plt.ylim(0, 1)
    plt.subplot(313)
    rs['accuracy'] = np.cumsum(rs['correct'])/np.cumsum(rs['predict_bool'])
    # rs['accuracy'] = np.cumsum((rs['sig']>0).astype(int))/np.cumsum((rs['sig']!=0).astype(int))
    # print 'the last accuracy:', rs['accuracy'][-1]
    plt.plot(rs['accuracy'], label='$accuracy$')
    plt.legend()
    plt.ylabel('accuracy')
    plt.ylim(0.4, 0.8)
    plt.figtext(0.15, 0.95, 'predict times:{}'.format(sum(rs['predict_bool'])), color='green')
    plt.figtext(0.4, 0.95, 'mdd num:{:.2f}'.format(max_drawdown_num(rs['sigsum'])), color='green')
    # sigsum_threshold = 200
    # plt.figtext(0.7, 0.95, 'mdd rate:{:.2f}%'.format(max_drawdown_rate(rs['sigsum'][rs['sigsum']>sigsum_threshold])*100), color='green')
    plt.figtext(0.15, 0.91, 'last sigsum:{:.1f}'.format(rs['sigsum'][-1]), color='green')
    plt.figtext(0.4, 0.91, 'last averagesig:{:.3f}'.format(rs['averagesig'][-1]), color='green')
    plt.figtext(0.7, 0.91, 'last accuracy:{:.2f}%'.format(rs['accuracy'][-1]*100), color='green')
    fig_dir = os.path.split(op.pn_out)[0]
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(op.pn_out)
    plt.close()
    # print rs
    #######################################################################################
    # ## resample siglist for specified periods
    # sig_resample = pd.DataFrame()
    # resample_time = '1D' ## you can change the resample time, for example, 5Min, 1D, 1W, 1M
    # sig_resample['sig'] = rs.sig.resample(resample_time, how='sum').fillna(0)
    # print sig_resample
    # ## plot
    # plt.figure()
    # plt.plot(sig_resample['sig'], 'k.', label='$sig$')
    # acc = sum((sig_resample['sig']>0).astype(int))/sum((sig_resample['sig']!=0).astype(int))
    # print 'accuracy(only for sig!=0):{:.2f}%'.format(acc*100)
    # plt.figtext(0.35, 0.93, 'accuracy(only for sig!=0):{:.2f}%'.format(acc*100), color='green')
    # plt.savefig('sig_{}.png'.format(resample_time))
    # plt.close()

def closeplot(op):
    csv_path = op.fn_in
    lines = sum(1 for _ in csv.reader(open(csv_path)))
    resample_time = 1
    data = loadTimeSeries(csv_path, rs_num = lines, nrows = lines, resample_time = resample_time)
    plt.plot(data.close)
    plt.savefig('close.png')

def sigingknn_talib(params):
    #######################################################################################
    starttime = datetime.datetime.now()
    csv_path, rs_num, nrows, resample_time, diff_n = params

    #######################################################################################
    '''1. DATA PREPARING'''
    #######################################################################################
    # rs_num, nrows = 10000, 10000
    data = loadTimeSeries(csv_path, rs_num = rs_num, nrows = nrows, resample_time = resample_time)
    ## ------------------------------------------------------------------------------------
    assert diff_n % resample_time == 0
    data['diff'] = data.close.diff(periods=int(diff_n/resample_time)).shift(-int(diff_n/resample_time)).fillna(0)
    zero_propotion = 0
    threshold = threshold_define(data, zero_propotion)
    data = target_define(data, threshold)
    # print data.head(10)
    #######################################################################################
    '''2. FEATURE EXTRACTION'''
    #######################################################################################
    X = feature_gen_talib(data, termlist = getFibonacciList(144))
    # X = getTermFeatures(data, windowsize = 17, termlist = getFibonacciList(144))
    ## ------------------------------------------------------------------------------------
    ## TODO(here you can do feature selection)
    # my_pca = PCA(n_components = 100).fit(X)
    # X = my_pca.transform(X)
    index = X.index
    #######################################################################################
    # '''change definition of 'diff', use relative close change instead of absolute close change'''
    # data['diff'] = data['diff']/data['close'] ## (close2-close1)/close1
    yset, closeset, diffset = data['label'].ix[index], data['close'].ix[index], data['diff'].ix[index]
    #######################################################################################
    window_size = int(len(index))
    logger.info('sigingknn_talib >> window_size:{}'.format(window_size))
    yset_window = yset.ix[-window_size:]
    closeset_window = closeset.ix[-window_size:]
    diffset_window = diffset.ix[-window_size:]
    ## ------------------------------------------------------------------------------------
    test_size = int(diff_n/resample_time)
    y_train = yset_window.ix[:-test_size]
    y_test = yset_window.ix[-test_size:]
    diff_train = diffset_window.ix[:-test_size]
    test_index = closeset_window.index[-test_size:]
    test_close = closeset_window.ix[-test_size:]
    #######################################################################################
    '''3. CLASSIFICATION'''
    #######################################################################################
    dataset_window = X.ix[-window_size:]
    X_train = dataset_window.ix[:-test_size]
    X_test = dataset_window.ix[-test_size:]
    y_pred = knn_classification(X_train, X_test, y_train, diff_train)
    #######################################################################################
    y_pred = y_pred.astype(int)
    ## no transaction in the final diff_n minutes of trading
    assert len(test_index) == len(y_pred)
    for i in range(len(test_index)):
        if isDayAcross(test_index[i], diff_n):
            y_pred[i] = 0
        else:
            pass
    logger.info('sigingknn_talib >> {}, {}, {}'.format(test_index[-1], y_pred[-1], test_close[-1]))

    #######################################################################################
    endtime = datetime.datetime.now()
    logger.info('sigingknn_talib >> consuming time:{}'.format(endtime-starttime))

    return pd.DataFrame({'predict': y_pred[-1], 'close':test_close[-1]}, index=[test_index[-1]])

def sigingknn(params):
    #######################################################################################
    starttime = datetime.datetime.now()
    csv_path, rs_num, nrows, resample_time, diff_n = params

    #######################################################################################
    '''1. DATA PREPARING'''
    #######################################################################################
    # rs_num, nrows = 10000, 10000
    data = loadTimeSeries(csv_path, rs_num = rs_num, nrows = nrows, resample_time = resample_time)
    ## ------------------------------------------------------------------------------------
    assert diff_n % resample_time == 0
    data['diff'] = data.close.diff(periods=int(diff_n/resample_time)).shift(-int(diff_n/resample_time)).fillna(0)
    zero_propotion = 0
    threshold = threshold_define(data, zero_propotion)
    data = target_define(data, threshold)
    # print data.head(10)
    #######################################################################################
    '''2. FEATURE EXTRACTION'''
    #######################################################################################
    X_ut = getUnitTermFeatures(data)
    X_st = getShortTermFeatures(data, windowsize = 17)
    X_lt = getLongTermFeatures(data, termlist = getFibonacciList(144))
    ## ------------------------------------------------------------------------------------
    '''you can change feature and index here'''
    index = sorted(list(set(X_ut.index) & set(X_st.index) & set(X_lt.index)))
    X_ut, X_st, X_lt = X_ut.ix[index], X_st.ix[index], X_lt.ix[index]
    #######################################################################################
    # '''change definition of 'diff', use relative close change instead of absolute close change'''
    # data['diff'] = data['diff']/data['close'] ## (close2-close1)/close1
    yset, closeset, diffset = data['label'].ix[index], data['close'].ix[index], data['diff'].ix[index]
    #######################################################################################
    window_size = int(len(index))
    logger.info('sigingknn >> window_size:{}'.format(window_size))
    yset_window = yset.ix[-window_size:]
    closeset_window = closeset.ix[-window_size:]
    diffset_window = diffset.ix[-window_size:]
    ## ------------------------------------------------------------------------------------
    test_size = int(diff_n/resample_time)
    y_train = yset_window.ix[:-test_size]
    y_test = yset_window.ix[-test_size:]
    diff_train = diffset_window.ix[:-test_size]
    test_index = closeset_window.index[-test_size:]
    test_close = closeset_window.ix[-test_size:]
    #######################################################################################
    '''3. CLASSIFICATION'''
    #######################################################################################
    '''multi_feature'''
    # X = pd.concat([X_ut, X_st, X_lt], axis=0)
    # dataset_window = X.ix[-window_size:]
    # X_train = dataset_window.ix[:-test_size]
    # X_test = dataset_window.ix[-test_size:]
    # y_pred = knn_classification(X_train, X_test, y_train, diff_train)
    #######################################################################################
    '''multi_classification'''
    X_ut_train = X_ut.ix[-window_size:].ix[:-test_size]
    X_ut_test = X_ut.ix[-window_size:].ix[-test_size:]
    y_pred_ut = knn_classification(X_ut_train, X_ut_test, y_train, diff_train)
    X_st_train = X_st.ix[-window_size:].ix[:-test_size]
    X_st_test = X_st.ix[-window_size:].ix[-test_size:]
    y_pred_st = knn_classification(X_st_train, X_st_test, y_train, diff_train)
    X_lt_train = X_lt.ix[-window_size:].ix[:-test_size]
    X_lt_test = X_lt.ix[-window_size:].ix[-test_size:]
    y_pred_lt = knn_classification(X_lt_train, X_lt_test, y_train, diff_train)
    ## ------------------------------------------------------------------------------------
    '''you can change classification here'''
    ## ------------------ one classifiers: -----------------------------------------------
    # y_pred = y_pred_st
    ## ------------------ two classifiers: just calculate sum ----------------------------
    # y_pred = np.sign(y_pred_st+y_pred_lt)
    y_pred = np.sign(y_pred_st+y_pred_lt)*(1-np.sign(np.abs(y_pred_st-y_pred_lt)))
    ## ------------------ three or more classification: Vote -----------------------------
    # y_pred = Vote(np.array([y_pred_ut, y_pred_st, y_pred_lt]).T)
    # y_pred = np.sign(y_pred_ut+y_pred_st+y_pred_lt)*\
    #          (1-np.sign(np.abs(y_pred_st-y_pred_lt)+np.abs(y_pred_lt-y_pred_ut)+np.abs(y_pred_ut-y_pred_st)))
    #######################################################################################
    y_pred = y_pred.astype(int)
    ## no transaction in the final diff_n minutes of trading
    assert len(test_index) == len(y_pred)
    for i in range(len(test_index)):
        if isDayAcross(test_index[i], diff_n):
            y_pred[i] = 0
        else:
            pass
    logger.info('sigingknn >> {}, {}, {}'.format(test_index[-1], y_pred[-1], test_close[-1]))

    #######################################################################################
    endtime = datetime.datetime.now()
    logger.info('sigingknn >> consuming time:{}'.format(endtime-starttime))

    return pd.DataFrame({'predict': y_pred[-1], 'close':test_close[-1]}, index=[test_index[-1]])

def sigingsvm(params):
    #######################################################################################
    starttime = datetime.datetime.now()
    csv_path, rs_num, nrows, resample_time, diff_n = params

    #######################################################################################
    '''1. DATA PREPARING'''
    #######################################################################################
    # rs_num, nrows = 10000, 10000
    data = loadTimeSeries(csv_path, rs_num = rs_num, nrows = nrows, resample_time = resample_time)
    ## ------------------------------------------------------------------------------------
    assert diff_n % resample_time == 0
    data['diff'] = data.close.diff(periods=int(diff_n/resample_time)).shift(-int(diff_n/resample_time)).fillna(0)
    zero_propotion = 0.06
    threshold = threshold_define(data, zero_propotion)
    data = target_define(data, threshold)
    # print data.head(10)
    #######################################################################################
    '''2. FEATURE EXTRACTION'''
    #######################################################################################
    X = feature_gen(data)
    X = X.drop(['open', 'high', 'low', 'close', 'volume', 'pos', 'neg', 'diff', 'label'], axis = 1)
    index = X.index
    #######################################################################################
    # '''change definition of 'diff', use relative close change instead of absolute close change'''
    # data['diff'] = data['diff']/data['close'] ## (close2-close1)/close1
    yset, closeset, diffset = data['label'].ix[index], data['close'].ix[index], data['diff'].ix[index]
    #######################################################################################
    window_size = int(len(index))
    logger.info('sigingsvm >> window_size:{}'.format(window_size))
    yset_window = yset.ix[-window_size:]
    closeset_window = closeset.ix[-window_size:]
    diffset_window = diffset.ix[-window_size:]
    ## ------------------------------------------------------------------------------------
    test_size = int(diff_n/resample_time)
    y_train = yset_window.ix[:-test_size]
    y_test = yset_window.ix[-test_size:]
    diff_train = diffset_window.ix[:-test_size]
    test_index = closeset_window.index[-test_size:]
    test_close = closeset_window.ix[-test_size:]
    #######################################################################################
    '''3. CLASSIFICATION'''
    #######################################################################################
    dataset_window = X.ix[-window_size:]
    X_train = dataset_window.ix[:-test_size]
    X_test = dataset_window.ix[-test_size:]
    y_pred = svm_classification(X_train, X_test, y_train, diff_train)

    #######################################################################################
    y_pred = y_pred.astype(int)
    ## no transaction in the final diff_n minutes of trading
    assert len(test_index) == len(y_pred)
    for i in range(len(test_index)):
        if isDayAcross(test_index[i], diff_n):
            y_pred[i] = 0
        else:
            pass
    logger.info('sigingsvm >> {}, {}, {}'.format(test_index[-1], y_pred[-1], test_close[-1]))

    #######################################################################################
    endtime = datetime.datetime.now()
    logger.info('sigingsvm >> consuming time:{}'.format(endtime-starttime))

    return pd.DataFrame({'predict': y_pred[-1], 'close':test_close[-1]}, index=[test_index[-1]])

def xx(op):
    classifier = op.classifier
    csv_path = op.fn_in
    diff_n = op.diff_n
    # resample_time = diff_n ## str or int, for example, '1min' or 1
    resample_time = 1 ## str or int, for example, '1min' or 1
    lines = sum(1 for _ in csv.reader(open(csv_path)))
    logger.info('xx >> {} lines in csv file'.format(lines))
    ## ------------------------------------------------------------------------------------
    if classifier == 'knn':
        nrows = int(lines*0.5) ## you can change the reading length here for forwardtest
    elif classifier == 'svm':
        nrows = min(int(lines*0.1), 800) ## you can change the reading length here for forwardtest
    elif classifier == 'knn_talib':
        nrows = int(lines*0.5) ## you can change the reading length here for forwardtest
    else:
        nrows = None
    ## ------------------------------------------------------------------------------------
    # predict_periods = 1000
    predict_periods = op.l_predict
    assert lines >= nrows+predict_periods
    step = diff_n ## because you predict label of the diff-th point in sliding window, you must use the next diff-th point to calculate diff and sigsum
    start = nrows
    end = nrows+predict_periods ## use 'end = nrows+1' just for the last point!!!
    rs_num_list = reversed(range(start, end, step))
    #### predict the last predict_periods periods, and predict int(1+(predict_periods-1)/step) points!!!
    logger.info('xx >> {} periods and {} points to predict'.format(predict_periods, int(1+(predict_periods-1)/step)))
    p = mp.Pool(op.n_job)
    ## ------------------------------------------------------------------------------------
    if classifier == 'knn':
        result = p.map(sigingknn, [(csv_path, rs_num, nrows, resample_time, diff_n) for rs_num in rs_num_list]) ## begin to read csv at 'lines-rs_num' line
    elif classifier == 'svm':
        result = p.map(sigingsvm, [(csv_path, rs_num, nrows, resample_time, diff_n) for rs_num in rs_num_list]) ## begin to read csv at 'lines-rs_num' line
    elif classifier == 'knn_talib':
        result = p.map(sigingknn_talib, [(csv_path, rs_num, nrows, resample_time, diff_n) for rs_num in rs_num_list]) ## begin to read csv at 'lines-rs_num' line
    else:
        result = []
    ## ------------------------------------------------------------------------------------
    results = pd.concat([i for i in result if type(i) == pd.DataFrame])
    results = results.reset_index().drop_duplicates(subset='index', take_last=True).set_index('index').sort()
    # print results
    results_dir = os.path.split(op.fn_out)[0]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results.to_csv(op.fn_out, index=1, sep='\t', header=1)
    #######################################################################################
    # resultplot(op)


if __name__ == '__main__':

    ## parse argument ##
    description = '' # A description of what the program does
    epilog = '' # Text following the argument descriptions
    parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', dest='fn_in', help='input file name.', default='./data/IF.csv') # change default
    parser.add_argument('-of', '--output_file', dest='fn_out', help='output file name.', default='./results/results.csv') # change default
    parser.add_argument('-op', '--output_pic', dest='pn_out', help='output pic name.', default='./results/results.png') # change default
    parser.add_argument('-nj', '--n_job', dest='n_job', help='job number.', type=int, default='1') # change default
    parser.add_argument('-lp', '--l_predict', dest='l_predict', help='predict periods.', type=int, default='10') # change default
    parser.add_argument('-dn', '--diff_n', dest='diff_n', help='diff_n number.', type=int, default='1') # change default
    parser.add_argument('-cf', '--classifier', dest='classifier', help='classifier.', default='knn') # change default
    op = parser.parse_args()

    ## record log:NOTSET<DEBUG<INFO<WARNING<ERROR<CRITICAL
    # logging.debug('debug')
    # logging.info('info')
    # logging.warn('warn')
    # logging.error('error')
    # logging.critical('critical')
    logger.info('main >> runing {} with parameters: {}'.format(APP_NAME, op))
    ## ------------------------------------------------------------------------------------
    '''Speed Up'''
    # xx(op)
    from speedtest import xxtest
    xxtest(op)
    # resultplot(op)
    # closeplot(op)


'''
optforwardtest system

command:
    python optforwardtest.py -i ./data/IF.csv -of ./results/results.csv -op ./results/results.png -nj 1 -lp 10 -dn 1 -cl knn

help:
    -i the input file
    -of the output file which records time index, predict label, and close
    -op the output file which describes sigsum and accuracy
    -nj number of jobs to run in parallel
    -lp length of periods to predict, not number of points to predict
    -dn length of periods to shift
    -cl classifier
'''
