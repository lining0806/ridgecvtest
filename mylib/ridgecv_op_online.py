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

##########################################################################################
class predict():

    def __init__(self, file_path='IF.csv'):
        self.file_path = file_path
        self.skip_rows = 1200000
        self.nrows = 1200000
        assert self.skip_rows >= self.nrows
        self.data = None
        self.features = None

        self.timeshift = 1
        self.resample = 1
        assert self.timeshift >= self.resample
        self.maxlag = 17
        self.indices = None
        self.cross_predict_days = None
        self.cross_predict_periods = None
        self.cross_predict_num = None
        self.cross_predict_reindex = 0
        self.windowsize = 300000
        self.threshold = 0

        self.Close_test = None
        self.X_test = None
        self.Close_train = None
        self.X_train = None
        self.y_train = None
        self.diff_train = None

        self.step = 4*2*60*60 # update model by step
        self.real_time_report = True
        #### you can change optunity target here!!!
        self.target = 'adjusted_sigsum' # 'accuracy', 'sigsum', 'sig_per_trade', 'adjusted_sigsum'

        self.y_pred = np.array([])
        self.y_targets = np.array([])
        self.diff_targets = np.array([])
        self.results_pred = []
        self.logs = True

        self.algorithm = 'ridgecv'

    def dataProcess(self, load_data=True):
        if load_data is True:
            self.data = self.loadTimeSeriesData() # Read in data
            ## ---------------------------------------------------------------------------------------
            # day_night_list = []
            # data_time = self.data.Time
            # for idx in xrange(self.data.shape[0]):
            #     if '09:00:00'<=data_time.iloc[idx]<='15:00:00':
            #         day_night_list.append('day')
            #     elif '21:00:00'<=data_time.iloc[idx]<='23:59:59' or '00:00:00'<=data_time.iloc[idx]<='01:00:00':
            #         day_night_list.append('night')
            #     else:
            #         day_night_list.append('')
            # self.data['day_night'] = day_night_list
            # data_day = self.data.ix[self.data['day_night']=='day']
            # data_night = self.data.ix[self.data['day_night']=='night']
            ## ---------------------------------------------------------------------------------------
            # self.removeDuplicate()
            # self.removeOutOfTradingtime()
            self.features = self.generateLagMatrix().astype(np.float16)
            #### save the data and features
            # with open('./data_feature', 'wb') as fp:
            #     pickle.dump((self.data, self.features), fp)
        else:
            if os.path.exists('./data_feature'):
                with open('./data_feature', 'rb') as fp:
                    self.data, self.features = pickle.load(fp)
            else:
                self.data, self.features = None, None

    def loadTimeSeriesData(self):
        assert self.skip_rows >= self.nrows
        def count_lines(filename):
            count = 0
            buffer_size = 1024*1024
            with open(filename,'rb') as f:
                while 1:
                    temp = f.read(buffer_size)
                    if not temp:
                        break
                    count += temp.count('\n')
            return count
        lines = count_lines(self.file_path)
        # lines = np.sum(1 for _ in csv.reader(open(self.file_path)))
        # print lines
        if lines < self.skip_rows:
            self.skip_rows = lines
        if lines < self.nrows:
            self.nrows = lines
        # data = pd.read_csv(self.file_path, header=None, engine='c')
        data = pd.read_csv(self.file_path,
                          engine='c',
                          header=None,
                          usecols=[0,1,2,4,6],
                          names=['Date','Time','Low','High','Close'],
                          skiprows=lines-self.skip_rows,
                          nrows=self.nrows,
        ) # read the latest nrows
        # print data.groupby(data.Date).size() # describe data size order by Date
        data.index = pd.to_datetime(data.Date+' '+data.Time+'.0',format='%Y-%m-%d %H:%M:%S.%f')
        # data.sort_index(axis=0, ascending=True, inplace=True) # Sort the data by time index
        data.drop_duplicates(keep='first', inplace=True) # Remove duplicate column
        # del data['Date'], data['Time']
        return data

    def loadTimeSeriesDataPro(self):
        assert self.skip_rows >= self.nrows
        def count_lines(filename):
            count = 0
            buffer_size = 1024*1024
            with open(filename,'rb') as f:
                while 1:
                    temp = f.read(buffer_size)
                    if not temp:
                        break
                    count += temp.count('\n')
            return count
        lines = count_lines(self.file_path)
        # lines = np.sum(1 for _ in csv.reader(open(self.file_path)))
        # print lines
        if lines < self.skip_rows:
            self.skip_rows = lines
        if lines < self.nrows:
            self.nrows = lines
        # data = pd.read_csv(self.file_path, header=None, engine='c')
        data = pd.read_csv(self.file_path,
                          engine='c',
                          header=None,
                          usecols=[0,1,2,3,4,5,6],
                          names=['Date','Time','Open','High','Low','Close','Volume'],
                          skiprows=lines-self.skip_rows,
                          nrows=self.nrows,
        ) # read the latest nrows
        # print data.groupby(data.Date).size() # describe data size order by Date
        data.index = pd.to_datetime(data.Date+' '+data.Time+'.0',format='%Y-%m-%d %H:%M:%S.%f')
        # data.sort_index(axis=0, ascending=True, inplace=True) # Sort the data by time index
        data.drop_duplicates(keep='first', inplace=True) # Remove duplicate column
        # del data['Date'], data['Time']
        return data

    def removeDuplicate(self):
        uniques = np.unique(self.data.index, return_index=True)[1]
        uniques.sort()
        # print uniques

        #### calculate index count
        # new_uniques = np.append(uniques, len(self.data.index))
        # diff_new_uniques = np.diff(new_uniques)
        # # print diff_new_uniques
        # count1, count2, count_other = 0, 0, 0
        # for i, x in enumerate(diff_new_uniques):
        #     if x == 1:
        #         # print self.data.index[new_uniques[i]]
        #         count1 += 1
        #     elif x == 2:
        #         # print self.data.index[new_uniques[i]]
        #         count2 += 1
        #     else:
        #         count_other += 1
        # print count1, count2, count_other

        self.data['new_index'] = self.data.index+Milli(500)
        self.data['new_index'].ix[uniques] -= Milli(500)
        self.data.set_index('new_index', drop=True, append=False, inplace=True)
        uniques_ = np.unique(self.data.index, return_index=True)[1]
        uniques_.sort()
        self.data = self.data.ix[uniques_]
        '''
        self.data = self.data.ix[np.unique(self.data.index, return_index=True)[1]] # Remove duplicate data
        '''
        '''
        position_dict = {}
        k, last_index = 0, None
        for index in self.data.index:
            if index != last_index:
                position_dict[index] = k
            # else:
            #    print 'Same timestamp %s' % index
            k += 1
            last_index = index
        self.data = self.data.ix[np.sort(position_dict.values())]
        '''

    def removeOutOfTradingtime(self):

        def isInTradingTime(timestamp):
            trading_time = timestamp.time()
            starttime1 = datetime.time(9, 15)
            stoptime1 = datetime.time(11, 30)
            starttime2 = datetime.time(13, 0)
            stoptime2 = datetime.time(15, 15)
            if starttime1<=trading_time<=stoptime1 or starttime2<=trading_time<=stoptime2:
                return True
            else:
                return False

        index = [item for item in self.data.index if isInTradingTime(item)]
        self.data = self.data.ix[index]

    def generateLagMatrix(self):

        def Price_change_ratio(price1, price2):
            # delta = price1 / price2-1
            delta = price1-price2
            return delta

        self.data['Middle'] = (self.data['High']+self.data['Low'])/2.0

        features = pd.DataFrame(index=self.data.index)
        MAX = 100
        # for i_shift in np.arange(1, MAX+1, 1):
        #     features['Close_lag_' +str(i_shift)] = Price_change_ratio(self.data.Close.shift(i_shift), self.data.Close)
        #     features['Middle_lag_'+str(i_shift)] = Price_change_ratio(self.data.Middle.shift(i_shift), self.data.Middle)
        for i_shift in np.arange(1, MAX+1, 1):
            features['Close_lag_' +str(i_shift)] = Price_change_ratio(self.data.Close.shift(i_shift), self.data.Close.shift(i_shift-1))
            features['Middle_lag_'+str(i_shift)] = Price_change_ratio(self.data.Middle.shift(i_shift), self.data.Middle.shift(i_shift-1))

        # features = features.fillna(0)
        features = features.drop(features.index[:MAX+1])

        self.data = self.data.drop(self.data.index[:MAX+1])

        return features

    def selectFeatures(self):
        self.features = self.features.ix[:, :self.maxlag*2]

    def targetDefine(self, threshold=0.0):
        self.data['Diff'] = self.data.Close.diff(self.timeshift).shift(-self.timeshift).fillna(0)
        pos = pd.Series(self.data['Diff']>threshold).astype(int)
        neg = -pd.Series(self.data['Diff']<-threshold).astype(int)
        self.data['Label'] = pos+neg
        self.data['Diff_'] = self.data.Close.diff(self.resample).shift(-self.resample).fillna(0)
        pos_ = pd.Series(self.data['Diff_']>threshold).astype(int)
        neg_ = -pd.Series(self.data['Diff_']<-threshold).astype(int)
        self.data['Label_'] = pos_+neg_

    def run(self, mode='test', online=True): ##########################################################################################

        if mode == 'forward':

            i_shift = 0
            self.indices = []
            if self.cross_predict_days is not None:
                self.daysToIndices(i_shift)
            elif self.cross_predict_periods is not None:
                self.periodsToIndices(i_shift)
            elif self.cross_predict_num is not None:
                self.numToIndices(i_shift)
            else:
                pass

            self.cross_predict_num = len(self.indices)

            print 'cross_predict_num:', self.cross_predict_num
            y_pred_array = np.zeros(self.cross_predict_num)
            y_targets_array = np.zeros(self.cross_predict_num)
            diff_targets_array = np.zeros(self.cross_predict_num)
            last_close_test, last_y_diff_pred, last_y_pred = 0, 0, 0

            Regression = None
            local_threshold = 0
            for i_cross_predict in np.arange(self.cross_predict_num):

                #### the test sets
                test_index = self.indices[i_cross_predict]
                # print test_index
                self.Close_test = self.data.Close.ix[test_index]
                self.X_test = self.features.ix[test_index]
                y_target = self.data.Label_.ix[test_index]
                diff_target = self.data.Diff_.ix[test_index]

                #### the train sets
                if i_cross_predict%self.step == 0:
                    start_index = test_index-self.windowsize-self.timeshift+1
                    end_index = test_index-self.timeshift+1
                    # print start_index
                    # print end_index
                    if start_index < 0:
                        start_index = 0
                    self.Close_train = self.data.Close.ix[start_index:end_index]
                    self.X_train = self.features.ix[start_index:end_index]
                    self.y_train = self.data.Label.ix[start_index:end_index]
                    self.diff_train = self.data.Diff.ix[start_index:end_index]
                    if self.algorithm == 'ridgecv':
                        Regression = RidgeCV().fit(self.X_train, self.diff_train)
                    elif self.algorithm == 'elasticnetcv':
                        Regression = ElasticNetCV().fit(self.X_train, self.diff_train)
                    elif self.algorithm == 'knnreg':
                        k = 200
                        Regression = KNeighborsRegressor(algorithm='auto', n_neighbors=k).fit(self.X_train, self.diff_train)
                    elif self.algorithm == 'linearsvr':
                        Regression = LinearSVR().fit(self.X_train, self.diff_train)
                        # Regression = SVR(kernel='linear').fit(self.X_train, self.diff_train)
                    else:
                        Regression = None
                    # abs_diffs = np.sort(np.abs(self.diff_train))
                    X_diff_pred = Regression.predict(self.X_train)
                    abs_diffs = np.sort(np.abs(X_diff_pred))
                    if self.threshold == 0:
                        local_threshold = 0
                    elif self.threshold == 100:
                        local_threshold = abs_diffs[-1]
                    else:
                        threshold_index = int(len(abs_diffs)*self.threshold*0.01)
                        local_threshold = abs_diffs[threshold_index]

                #### prediction stage
                y_diff_pred = Regression.predict(self.X_test.reshape(1, -1))[-1] # Only one predict in the result
                if np.abs(y_diff_pred) < local_threshold:
                    y_pred = 0
                elif y_diff_pred > 0.0:
                    y_pred = 1
                elif y_diff_pred < 0.0:
                    y_pred = -1
                else:
                    y_pred = 0
                ##########################################################################################
                close_test = self.Close_test
                if online and i_cross_predict > 0:
                    y_diff_pred += last_close_test*last_y_diff_pred/close_test
                    y_diff_pred /= 2.0
                if np.abs(y_diff_pred) < local_threshold:
                    y_pred = 0
                elif y_diff_pred > 0.0:
                    y_pred = 1
                elif y_diff_pred < 0.0:
                    y_pred = -1
                else:
                    y_pred = 0

                # close_test = self.Close_test
                # if online and i_cross_predict > 0:
                #     if np.abs(last_y_pred-y_pred) <= 1:
                #         pass
                #     else:
                #         y_pred = 0
                ##########################################################################################
                # print 'Last point:%s close: %6.1f Prediction for next point: %2d' % (self.X_test.name, self.Close_test, y_pred)
                if self.logs:
                    self.results_pred.append('%s\t%2d' % (self.X_test.name, y_pred))

                y_pred_array[i_cross_predict] = y_pred
                y_targets_array[i_cross_predict] = y_target
                diff_targets_array[i_cross_predict] = diff_target
                last_close_test, last_y_diff_pred, last_y_pred = close_test, y_diff_pred, y_pred

            #### describe
            accuracy_list, sigsum, real_sigsum, adjusted_sigsum, trade_count, sig_per_trade = self.resultsDescribe(y_pred_array, y_targets_array, diff_targets_array)

            self.y_pred = np.append(self.y_pred, y_pred_array)
            self.y_targets = np.append(self.y_targets, y_targets_array)
            self.diff_targets = np.append(self.diff_targets, diff_targets_array)

            last_accuracy = accuracy_list[-1]
            last_sigsum = sigsum[-1]
            last_adjusted_sigsum = adjusted_sigsum[-1]
            last_trade_count = trade_count[-1]
            last_sig_per_trade = sig_per_trade[-1]

            return last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade

        elif mode == 'test':

            last_accuracy_list = np.array([])
            last_sigsum_list = np.array([])
            last_sig_per_trade_list = np.array([])
            last_adjusted_sigsum_list = np.array([])
            last_trade_count_list = np.array([])

            for i_shift in np.arange(self.resample):

                self.indices = []
                if self.cross_predict_days is not None:
                    self.daysToIndices(i_shift)
                elif self.cross_predict_periods is not None:
                    self.periodsToIndices(i_shift)
                elif self.cross_predict_num is not None:
                    self.numToIndices(i_shift)
                else:
                    pass

                self.cross_predict_num = len(self.indices)

                print 'cross_predict_num:', self.cross_predict_num
                y_pred_array = np.zeros(self.cross_predict_num)
                y_targets_array = np.zeros(self.cross_predict_num)
                diff_targets_array = np.zeros(self.cross_predict_num)
                last_close_test, last_y_diff_pred, last_y_pred = 0, 0, 0

                Regression = None
                local_threshold = 0
                for i_cross_predict in np.arange(self.cross_predict_num):

                    #### the test sets
                    test_index = self.indices[i_cross_predict]
                    # print test_index
                    self.Close_test = self.data.Close.ix[test_index]
                    self.X_test = self.features.ix[test_index]
                    y_target = self.data.Label_.ix[test_index]
                    diff_target = self.data.Diff_.ix[test_index]

                    #### the train sets
                    if i_cross_predict%self.step == 0:
                        start_index = test_index-self.windowsize-self.timeshift+1
                        end_index = test_index-self.timeshift+1
                        # print start_index
                        # print end_index
                        if start_index < 0:
                            start_index = 0
                        self.Close_train = self.data.Close.ix[start_index:end_index]
                        self.X_train = self.features.ix[start_index:end_index]
                        self.y_train = self.data.Label.ix[start_index:end_index]
                        self.diff_train = self.data.Diff.ix[start_index:end_index]
                        if self.algorithm == 'ridgecv':
                            Regression = RidgeCV().fit(self.X_train, self.diff_train)
                        elif self.algorithm == 'elasticnetcv':
                            Regression = ElasticNetCV().fit(self.X_train, self.diff_train)
                        elif self.algorithm == 'knnreg':
                            k = 200
                            Regression = KNeighborsRegressor(algorithm='auto', n_neighbors=k).fit(self.X_train, self.diff_train)
                        elif self.algorithm == 'linearsvr':
                            Regression = LinearSVR().fit(self.X_train, self.diff_train)
                            # Regression = SVR(kernel='linear').fit(self.X_train, self.diff_train)
                        else:
                            Regression = None
                        if self.threshold == 0:
                            local_threshold = 0
                        else:
                            X_diff_pred = Regression.predict(self.X_train)
                            abs_diffs = np.sort(np.abs(X_diff_pred))
                            threshold_index = int(len(abs_diffs)*self.threshold*0.01)
                            local_threshold = abs_diffs[threshold_index]
                            # abs_diffs = np.sort(np.abs(self.diff_train))
                            # threshold_index = int(len(abs_diffs)*self.threshold*0.01)
                            # local_threshold = abs_diffs[threshold_index]

                    #### prediction stage
                    y_diff_pred = Regression.predict(self.X_test.reshape(1, -1))[-1] # Only one predict in the result
                    if np.abs(y_diff_pred) < local_threshold:
                        y_pred = 0
                    elif y_diff_pred > 0.0:
                        y_pred = 1
                    elif y_diff_pred < 0.0:
                        y_pred = -1
                    else:
                        y_pred = 0
                    ##########################################################################################
                    close_test = self.Close_test
                    if online and i_cross_predict > 0:
                        y_diff_pred += last_close_test*last_y_diff_pred/close_test
                        y_diff_pred /= 2.0
                    if np.abs(y_diff_pred) < local_threshold:
                        y_pred = 0
                    elif y_diff_pred > 0.0:
                        y_pred = 1
                    elif y_diff_pred < 0.0:
                        y_pred = -1
                    else:
                        y_pred = 0

                    # close_test = self.Close_test
                    # if online and i_cross_predict > 0:
                    #     if np.abs(last_y_pred-y_pred) <= 1:
                    #         pass
                    #     else:
                    #         y_pred = 0
                    ##########################################################################################
                    # print 'Last point:%s close: %6.1f Prediction for next point: %2d' % (self.X_test.name, self.Close_test, y_pred)

                    y_pred_array[i_cross_predict] = y_pred
                    y_targets_array[i_cross_predict] = y_target
                    diff_targets_array[i_cross_predict] = diff_target
                    last_close_test, last_y_diff_pred, last_y_pred = close_test, y_diff_pred, y_pred

                #### describe
                accuracy_list, sigsum, real_sigsum, adjusted_sigsum, trade_count, sig_per_trade = self.resultsDescribe(y_pred_array, y_targets_array, diff_targets_array)

                last_accuracy_list = np.append(last_accuracy_list, accuracy_list[-1])
                last_sigsum_list = np.append(last_sigsum_list, sigsum[-1])
                last_adjusted_sigsum_list = np.append(last_adjusted_sigsum_list, adjusted_sigsum[-1])
                last_trade_count_list = np.append(last_trade_count_list, trade_count[-1])
                last_sig_per_trade_list = np.append(last_sig_per_trade_list, sig_per_trade[-1])

            #### calculate average
            last_accuracy = np.mean(last_accuracy_list)
            last_sigsum = np.mean(last_sigsum_list)
            last_adjusted_sigsum = np.mean(last_adjusted_sigsum_list)
            last_trade_count = np.mean(last_trade_count_list)
            last_sig_per_trade = np.mean(last_sig_per_trade_list)

            return last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade

        else:
            return None

    def daysToIndices(self, i_shift):
        assert self.timeshift >= self.resample
        date_list = np.sort(list(set(self.data.Date)))
        test_dates = date_list[-self.cross_predict_days:]
        # print test_dates
        # print len(test_dates)
        start = self.data[test_dates[0]].ix[0].name
        start_index = len(self.data.ix[:start])-1
        self.indices = np.arange(start_index-self.cross_predict_reindex+i_shift, len(self.data.index)-self.cross_predict_reindex, self.resample)

    def periodsToIndices(self, i_shift):
        assert self.timeshift >= self.resample
        start_index = len(self.data.index)-self.cross_predict_periods
        self.indices = np.arange(start_index-self.cross_predict_reindex+i_shift, len(self.data.index)-self.cross_predict_reindex, self.resample)

    def numToIndices(self, i_shift):
        assert self.timeshift >= self.resample
        start_index = len(self.data.index)-self.cross_predict_num*self.resample
        self.indices = np.arange(start_index-self.cross_predict_reindex+i_shift, len(self.data.index)-self.cross_predict_reindex, self.resample)

    def resultsDescribe(self, y_pred, y_targets, diff_targets):
        #### describe
        siglist = np.array(y_pred*diff_targets)
        sigsum = np.cumsum(siglist)
        # accuracy_list = self.cal_accuracy(siglist)
        # max_drawdown, mdd_duration = self.cal_maxDrawDown(sigsum)
        # str1 = 'accuracy: %.4f, max_drawdown: %.1f, mdd_duration: %d' % (accuracy_list[-1], max_drawdown, mdd_duration)
        # if self.real_time_report:
        #     print str1
        real_siglist, real_sigsum, trade_count = self.real_calculate(y_pred, diff_targets)
        accuracy_list = self.cal_accuracy(real_siglist)
        max_drawdown, mdd_duration = self.cal_maxDrawDown(real_sigsum)
        slide_penalty = 1
        adjusted_sigsum = real_sigsum-trade_count*slide_penalty
        max_drawdown, mdd_duration = self.cal_maxDrawDown(adjusted_sigsum)
        sig_per_trade = adjusted_sigsum*1.0/trade_count
        str2 = 'accuracy: %.4f, max_drawdown: %.1f, mdd_duration: %d' % (accuracy_list[-1], max_drawdown, mdd_duration)
        str3 = 'trade_count: %d, real_sigsum: %6.1f, adjusted_sigsum: %6.1f' % (trade_count[-1], real_sigsum[-1], adjusted_sigsum[-1])
        if self.real_time_report:
            print str2
            print str3
        return accuracy_list, sigsum, real_sigsum, adjusted_sigsum, trade_count, sig_per_trade

    @staticmethod
    def cal_accuracy(siglist):
        predict_bool = np.array(siglist!=0).astype(int)
        # correct_list = np.array(predict_bool&(siglist>=0)).astype(int)
        correct_list = np.array(siglist>0).astype(int)
        accuracy_list = np.cumsum(correct_list).astype(float)/np.cumsum(predict_bool)
        # last_accuracy = accuracy_list[-1]
        # last_accuracy = float(np.sum(correct_list))/np.sum(predict_bool)
        return accuracy_list

    @staticmethod
    def cal_maxDrawDown(sigsum):
        '''
        Calculate max drawn down within sigsum.
        Use numpy.maximum.accumulate to generate running maximum, then identifies the max drop
        Returns max drawdown in float
        '''
        bottom_index = np.argmax(np.maximum.accumulate(sigsum)-sigsum) # end of the period, the bottom
        # peak_index = np.argmax(sigsum[:bottom_index]) # start of period, the peak
        # max_drawdown = sigsum[peak_index]-sigsum[bottom_index]
        # mdd_duration = np.abs(bottom_index-peak_index)
        # return max_drawdown, mdd_duration
        if bottom_index == 0:
            return 0, 0
        else:
            peak_index = np.argmax(sigsum[:bottom_index]) # start of period, the peak
            max_drawdown = sigsum[peak_index]-sigsum[bottom_index]
            mdd_duration = np.abs(bottom_index-peak_index)
            return max_drawdown, mdd_duration

    # @staticmethod
    # def cal_maxDrawDown(sigsum):
    #     '''
    #     Return the absolute value of the maximum drawdown of sequence X.
    #
    #     Note
    #     ----
    #     If the sequence is strictly increasing, 0 is returned.
    #     '''
    #     peak = bottom = sigsum[0]
    #     peak_index = bottom_index = 0
    #     max_drawdown = 0
    #     for i, x in enumerate(sigsum):
    #         if x > peak:
    #             peak = x
    #             peak_index = i
    #         # drawdown = (peak - x) / peak
    #         drawdown = peak - x
    #         if drawdown > max_drawdown:
    #             max_drawdown = drawdown
    #             bottom = x
    #             bottom_index = i
    #     mdd_duration = bottom_index-peak_index
    #     return max_drawdown, mdd_duration

    @staticmethod
    def real_calculate(y_pred, diff_targets):
        df = pd.DataFrame()
        df['preds'] = y_pred
        df['diffs'] = diff_targets
        ## -------------------------------------------------------------------------------
        '''replace 0 with values before'''
        df['preds'] = df['preds'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        ## -------------------------------------------------------------------------------
        df['preds_turning'] = df['preds'].diff(1).replace(0, np.nan) # the first point is nan, and replace the unchange points with nan
        real_siglist = np.array(df['preds']*df['diffs'])
        real_sigsum = np.cumsum(real_siglist) # calculate real_sigsum
        trade_count = np.zeros(len(df['preds_turning']))
        for i in range(len(df['preds_turning'])):
            trade_count[i] = len(df['preds_turning'][:i+1].dropna())+1
        return real_siglist, real_sigsum, trade_count

    @staticmethod
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

##########################################################################################
def para_optunity(aa):

    def my_object(maxlag, windowsize, threshold):
        aa.maxlag = int(maxlag)
        aa.windowsize = int(windowsize)
        aa.threshold = threshold
        aa.selectFeatures()
        last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade = aa.run(mode='test')
        '''you can change target here'''
        if aa.target == 'accuracy':
            target = last_accuracy
        elif aa.target == 'sigsum':
            target = last_sigsum
        elif aa.target == 'sig_per_trade':
            target = last_sig_per_trade
        elif aa.target == 'adjusted_sigsum':
            target = last_adjusted_sigsum
        else:
            target = last_sigsum # default
        return target

    def my_object_algo(algorithm, maxlag, windowsize, threshold):
        aa.algorithm = algorithm
        aa.maxlag = int(maxlag)
        aa.windowsize = int(windowsize)
        aa.threshold = threshold
        aa.selectFeatures()
        last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade = aa.run(mode='test')
        '''you can change target here'''
        if aa.target == 'accuracy':
            target = last_accuracy
        elif aa.target == 'sigsum':
            target = last_sigsum
        elif aa.target == 'sig_per_trade':
            target = last_sig_per_trade
        elif aa.target == 'adjusted_sigsum':
            target = last_adjusted_sigsum
        else:
            target = last_sigsum # default
        return target

    ##########################################################################################
    '''
    PSO
    http://optunity.readthedocs.io/en/latest/_modules/optunity/solvers/ParticleSwarm.html#ParticleSwarm

    d = dict(kwargs)
    if num_evals > 1000:
        d['num_particles'] = 100
    elif num_evals >= 200:
        d['num_particles'] = 20
    elif num_evals >= 10:
        d['num_particles'] = 10
    else:
        d['num_particles'] = num_evals
    d['num_generations'] = int(math.ceil(float(num_evals) / d['num_particles']))
    return d
    '''
    maxlag = [30, 90]
    windowsize = [50000, 900000]
    threshold = [95, 100]
    search = {
        'algorithm':{'ridgecv':None},
        # 'algorithm':{'ridgecv':None,'elasticnetcv':None,'knnreg':None,'linearsvr':None},
        'maxlag':maxlag,
        'windowsize':windowsize,
        'threshold':threshold,
        }
    num_evals = 100

    ##################################################################################
    #### number_of_processes must equal to num_particles!!!
    if num_evals > 1000:
        number_of_processes = 100
    elif num_evals >= 500:
        number_of_processes = 50
    elif num_evals >= 300:
        number_of_processes = 30
    elif num_evals >= 100:
        number_of_processes = 20
    elif num_evals >= 30:
        number_of_processes = 10
    elif num_evals >= 10:
        number_of_processes = 5
    else:
        number_of_processes = num_evals
    ## -------------------------------------------------------------------------------
    #### ParticleSwarm_New
    from optunity.solvers.ParticleSwarm_New import ParticleSwarm_New
    best_params, info, _ = optunity.maximize(
    # best_params, info, _ = optunity.minimize(
        my_object,
        solver_name = 'particle swarm new', # default:'particle swarm'
        # solver_name = 'grid search', # default:'particle swarm'
        num_evals = num_evals,
        maxlag = maxlag,
        windowsize = windowsize,
        threshold = threshold,
        # pmap = optunity.pmap, # Parallel map using multiprocessing
        # pmap = pmap,
        pmap = create_pmap(number_of_processes),
    )
    # print info.optimum
    ## -------------------------------------------------------------------------------
    # #### ParticleSwarm
    # best_params, info, _ = optunity.maximize(
    # # best_params, info, _ = optunity.minimize(
    #     my_object,
    #     solver_name = 'particle swarm', # default:'particle swarm'
    #     # solver_name = 'grid search', # default:'particle swarm'
    #     num_evals = num_evals,
    #     maxlag = maxlag,
    #     windowsize = windowsize,
    #     threshold = threshold,
    #     # pmap = optunity.pmap, # Parallel map using multiprocessing
    #     # pmap = pmap,
    #     pmap = create_pmap(number_of_processes),
    # )
    # # print info.optimum
    ## -------------------------------------------------------------------------------
    # #### ParticleSwarm
    # best_params, info, _ = optunity.maximize_structured( # default:'particle swarm'
    # # best_params, info, _ = optunity.minimize_structured( # default:'particle swarm'
    #     my_object_algo,
    #     search_space=search,
    #     num_evals = num_evals,
    #     # pmap = optunity.pmap, # Parallel map using multiprocessing
    #     # pmap = pmap,
    #     pmap = create_pmap(number_of_processes),
    # )
    # # print info.optimum

    ##################################################################################
    df = optunity.call_log2dataframe(info.call_log)
    df.sort_values('value', ascending=False, inplace=True)

    return best_params, info.optimum, df

def _fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        value = f(*x)
        if hasattr(f, 'call_log'):
            k = list(f.call_log.keys())[-1]
            q_out.put((i, value, k))
        else:
            q_out.put((i, value))

# http://stackoverflow.com/a/16071616
def pmap(f, *args, **kwargs):
    """Parallel map using multiprocessing.

    :param f: the callable
    :param args: arguments to f, as iterables
    :returns: a list containing the results

    .. warning::
        This function will not work in IPython: https://github.com/claesenm/optunity/issues/8.

    .. warning::
        Python's multiprocessing library is incompatible with Jython.

    """
    nprocs = kwargs.get('number_of_processes', multiprocessing.cpu_count())
    # nprocs = multiprocessing.cpu_count()
    q_in = multiprocessing.Queue(1) # q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=_fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        '''
        Some threads do background tasks,
        like sending keepalive packets,
        or performing periodic garbage collection,
        or whatever.
        These are only useful when the main program is running,
        and it's okay to kill them off once the other, non-daemon, threads have exited.
        Without daemon threads, you'd have to keep track of them,
        and tell them to exit, before your program can completely quit.
        By setting them as daemon threads, you can let them run and forget about them,
        and when your program quits, any daemon threads are killed automatically.
        '''
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(zip(*args))]
    ##########################################################################################
    ## best way
    # [q_in.put((None, None)) for _ in range(nprocs)]
    # res = [q_out.get() for _ in range(len(sent))]
    ## ---------------------------------------------------------------------------------------
    ## best way
    res = [q_out.get() for _ in range(len(sent))]
    [q_in.put((None, None)) for _ in range(nprocs)]
    ##########################################################################################
    # for p in proc:
    #     p.terminate()
    ##########################################################################################
    for p in proc:
        p.join()

    # FIXME: strong coupling between pmap and functions.logged
    if hasattr(f, 'call_log'):
        for _, value, k in sorted(res):
            f.call_log[k] = value
        return [x for i, x, _ in sorted(res)]
    else:
        return [x for i, x in sorted(res)]

def create_pmap(number_of_processes):
    def pmap_bound(f, *args):
        return pmap(f, *args, number_of_processes=number_of_processes)
    return pmap_bound


##########################################################################################
if __name__ == '__main__':

    start_time = datetime.datetime.now()

    ##########################################################################################
    # file_path = './data/rb.csv'
    # a = predict(file_path=file_path)
    # # a = predict(file_path='rb1610_tick.csv')
    # # a = predict(file_path='/opt/share/rb1610_tick.csv')
    # # a = predict(file_path='./data/rb1610_tick.csv')
    # a.real_time_report = True
    #
    # ## ---------------------------------------------------------------------------------------
    # a.skip_rows = 1200000
    # a.dataProcess(load_data=True)
    #
    # a.timeshift = 1
    # # a.resample = a.timeshift
    # a.resample = 1
    # a.targetDefine()
    #
    # # a.cross_predict_days = 1
    # a.cross_predict_periods = 4*2*60*60
    # # a.cross_predict_num = 4*2*60*60
    #
    # a.algorithm = 'ridgecv'
    # a.maxlag = 40 ## <=100
    # a.windowsize = 300000
    # a.threshold = 90
    #
    # a.selectFeatures()
    # last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade = a.run(mode='forward')
    # print 'accuracy: %.4f, trade_count: %d, sigsum: %6.1f, adjusted_sigsum: %6.1f' % \
    #        (last_accuracy, last_trade_count, last_sigsum, last_adjusted_sigsum)
    #
    # accuracy_list, sigsum, real_sigsum, adjusted_sigsum, trade_count, sig_per_trade = a.resultsDescribe(a.y_pred, a.y_targets, a.diff_targets)
    # results_path = 'results'
    # if os.path.exists(results_path):
    #     shutil.rmtree(results_path)
    # if not os.path.exists(results_path):
    #     os.makedirs(results_path)
    # with open(os.path.join(results_path, 'results.pkl'), 'wb') as fp:
    #     pickle.dump((accuracy_list, sigsum, real_sigsum, adjusted_sigsum), fp)
    # print 'accuracy: %.4f, trade_count: %d, sigsum: %6.1f, real_sigsum: %6.1f, adjusted_sigsum: %6.1f' % \
    #        (accuracy_list[-1], trade_count[-1], sigsum[-1], real_sigsum[-1], adjusted_sigsum[-1])
    # a.pp(os.path.join(results_path, 'accuracy_sigsum_tradecount%d.png' % trade_count[-1]), accuracy_list, sigsum, real_sigsum, adjusted_sigsum)

    ##########################################################################################
    file_path = './data/rb.csv'

    ## ---------------------------------------------------------------------------------------
    #### calculate data size according to date
    temp = predict(file_path=file_path)
    # temp = predict(file_path='rb1610_tick.csv')
    # temp = predict(file_path='/opt/share/rb1610_tick.csv')
    # temp = predict(file_path='./data/rb1610_tick.csv')
    temp.real_time_report = False
    temp.skip_rows = 2000000
    temp.nrows = 2000000
    temp.dataProcess(load_data=True)
    print temp.data.groupby(temp.data.Date).size() # describe data size order by Date
    print temp.data.groupby(temp.data.Date).size()['2016-04-01':'2016-04-15']
    size1 = np.sum(temp.data.groupby(temp.data.Date).size()['2016-03-01':])
    size2 = np.sum(temp.data.groupby(temp.data.Date).size()['2016-06-01':])
    # size2 = 0
    print size1, size2

    del temp
    gc.collect()

    ####
    # size1 = 4*2*60*60
    # size2 = 0

    ## ---------------------------------------------------------------------------------------
    resample_num_list = [1]
    # resample_num_list = [1, 2, 4, 8]
    # resample_num_list = [1, 2, 3, 4, 5, 6, 7, 8]
    timeshift_num_list = [1]
    # timeshift_num_list = [1, 2, 4, 8, 16]
    # timeshift_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]

    # results_dir = './'
    results_dir = './results_dir/'
    default_nrows = 1200000

    for resample_num in resample_num_list:

        '''you can change the condition here: whether timeshift equals resample or not'''
        # timeshift_num_list = [resample_num] ## timeshift == resample
        timeshift_num_list = filter(lambda x: x>=resample_num, timeshift_num_list) ## timeshift >= resample

        for timeshift_num in timeshift_num_list:
            ## ---------------------------------------------------------------------------------------
            # #### calculate the first best parameter sets
            # temp = predict(file_path=file_path)
            # # temp = predict(file_path='rb1610_tick.csv')
            # # temp = predict(file_path='/opt/share/rb1610_tick.csv')
            # # temp = predict(file_path='./data/rb1610_tick.csv')
            # temp.real_time_report = False
            #
            # num = 4*2*60*60
            # temp.skip_rows = default_nrows+size1
            # temp.dataProcess(load_data=True)
            #
            # temp.timeshift = timeshift_num
            # # temp.resample = temp.timeshift
            # temp.resample = resample_num
            # temp.targetDefine()
            #
            # # temp.cross_predict_days = 1
            # temp.cross_predict_periods = num
            # # temp.cross_predict_num = num
            #
            # best_params, optimum, df_sort = para_optunity(temp)
            # print 'best parameters:', best_params
            # print 'best score:', optimum
            # print 'sorted best parameters:'
            # print df_sort
            #
            # del temp
            # gc.collect()
            #
            # save_path = './best_parameters'
            # with open(save_path, 'wb') as fp:
            #     pickle.dump((best_params, optimum, df_sort), fp)

            ####
            save_path = './best_parameters'
            with open(save_path, 'rb') as fp:
                best_params, optimum, df_sort = pickle.load(fp)
            # best_params = {
            #     'algorithm':'ridgecv',
            #     'maxlag':40, ## <=100
            #     'windowsize':300000,
            #     'threshold':30
            # }

            ## ---------------------------------------------------------------------------------------
            #### Dynamic parameter optimization
            dynamic = False

            a = predict(file_path=file_path)
            # a = predict(file_path='rb1610_tick.csv')
            # a = predict(file_path='/opt/share/rb1610_tick.csv')
            # a = predict(file_path='./data/rb1610_tick.csv')
            a.real_time_report = True

            num = 4*2*60*60
            data_file = './Data.h5'
            features_file = './Feature.h5'
            if os.path.exists(data_file):
                os.remove(data_file)
            if os.path.exists(features_file):
                os.remove(features_file)
            old_data = None
            old_features = None
            best_parameters_path = results_dir+'best_parameters_'+'resample'+str(resample_num)+'timeshift'+str(timeshift_num)
            if os.path.exists(best_parameters_path):
                shutil.rmtree(best_parameters_path)
            skip_rows_list = range(default_nrows+size1, default_nrows+size2, -num)+[default_nrows+size2]
            cross_predict_list = [skip_rows_list[i]-skip_rows_list[i+1] for i in range(len(skip_rows_list)-1)]
            for i_cross_predict_list in range(0, len(cross_predict_list)):
                skip_rows = skip_rows_list[i_cross_predict_list+1]
                num = cross_predict_list[i_cross_predict_list]
                ## ---------------------------------------------------------------------------------------
                #### add hdf5 cache
                # if os.path.exists(data_file) and os.path.exists(features_file):
                #     old_data = pd.read_hdf(data_file, 'df')
                #     old_features = pd.read_hdf(features_file, 'df')
                #     temp = predict(file_path=file_path)
                #     # temp = predict(file_path='rb1610_tick.csv')
                #     # temp = predict(file_path='/opt/share/rb1610_tick.csv')
                #     # temp = predict(file_path='./data/rb1610_tick.csv')
                #     temp.real_time_report = False
                #     temp.skip_rows = skip_rows-default_nrows+2*num
                #     temp.nrows = 2*num
                #     temp.dataProcess(load_data=True)
                #     a.data = pd.concat([old_data.ix[num:], temp.data.ix[-num:]])
                #     a.features = pd.concat([old_features.ix[num:], temp.features.ix[-num:]])
                #     del old_data, old_features
                #     del temp
                #     gc.collect()
                # else:
                #     a.skip_rows = skip_rows
                #     a.dataProcess(load_data=True)
                # a.data.to_hdf(data_file, 'df')
                # a.features.to_hdf(features_file, 'df')

                #### add memory cache
                if old_data is not None and old_features is not None:
                    temp = predict(file_path=file_path)
                    # temp = predict(file_path='rb1610_tick.csv')
                    # temp = predict(file_path='/opt/share/rb1610_tick.csv')
                    # temp = predict(file_path='./data/rb1610_tick.csv')
                    temp.real_time_report = False
                    temp.skip_rows = skip_rows-default_nrows+2*num
                    temp.nrows = 2*num
                    temp.dataProcess(load_data=True)
                    a.data = pd.concat([old_data.ix[num:], temp.data.ix[-num:]])
                    a.features = pd.concat([old_features.ix[num:], temp.features.ix[-num:]])
                    del old_data, old_features
                    del temp
                    gc.collect()
                else:
                    a.skip_rows = skip_rows
                    a.dataProcess(load_data=True)
                #### using deepcopy
                '''wrong'''
                # old_data = a.data
                # old_features = a.features
                '''right'''
                old_data = copy.deepcopy(a.data)
                old_features = copy.deepcopy(a.features)

                #### without cache
                # a.skip_rows = skip_rows
                # a.dataProcess(load_data=True)

                ## ---------------------------------------------------------------------------------------
                a.timeshift = timeshift_num
                # a.resample = a.timeshift
                a.resample = resample_num
                a.targetDefine()

                # a.cross_predict_days = 1
                a.cross_predict_periods = num
                # a.cross_predict_num = num

                # last_best_params = best_params
                last_df_sort = df_sort
                #### calculate the next best parameter sets
                if dynamic is True:
                    ## ---------------------------------------------------------------------------------------
                    #### using deepcopy
                    temp = copy.deepcopy(a)

                    ## ---------------------------------------------------------------------------------------
                    best_params, optimum, df_sort = para_optunity(temp)
                    print 'best parameters:', best_params
                    print 'best score:', optimum
                    print 'sorted best parameters:'
                    print df_sort

                    if not os.path.exists(best_parameters_path):
                        os.makedirs(best_parameters_path)
                    save_path = os.path.join(best_parameters_path, 'best_parameters_'+str(skip_rows))
                    with open(save_path, 'wb') as fp:
                        pickle.dump((best_params, optimum, df_sort), fp)
                    #### plot parameters distribution
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
                    # ####
                    # pic_path = os.path.join(best_parameters_path, 'best_parameters_'+str(skip_rows)+'.png')
                    # plt.savefig(pic_path)
                    # plt.close(fig)

                    ## ---------------------------------------------------------------------------------------
                    del temp
                    gc.collect()

                ## ---------------------------------------------------------------------------------------
                #### using the best parameter sets for prediction
                last_best_params = last_df_sort.iloc[0].to_dict()
                if 'algorithm' in last_best_params:
                    a.algorithm = last_best_params['algorithm']
                a.maxlag = int(last_best_params['maxlag'])
                a.windowsize = int(last_best_params['windowsize'])
                a.threshold = last_best_params['threshold']
                a.selectFeatures()
                last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade = a.run(mode='forward')

                ## ---------------------------------------------------------------------------------------
                # df_sort_test = []
                # for i_best_params in range(5):
                #     last_best_params = last_df_sort.iloc[i_best_params].to_dict()
                #     # last_best_params.pop('value')
                #     temp = copy.deepcopy(a)
                #     if 'algorithm' in last_best_params:
                #         temp.algorithm = last_best_params['algorithm']
                #     temp.maxlag = int(last_best_params['maxlag'])
                #     temp.windowsize = int(last_best_params['windowsize'])
                #     temp.threshold = last_best_params['threshold']
                #     temp.selectFeatures()
                #     last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade = temp.run(mode='forward')
                #     '''you can change target here'''
                #     if temp.target == 'accuracy':
                #         target = last_accuracy
                #     elif temp.target == 'sigsum':
                #         target = last_sigsum
                #     elif temp.target == 'sig_per_trade':
                #         target = last_sig_per_trade
                #     elif temp.target == 'adjusted_sigsum':
                #         target = last_adjusted_sigsum
                #     else:
                #         target = last_sigsum # default
                #     df_sort_test.append((target, last_best_params))
                #     del temp
                #     gc.collect()
                # df_sort_test.sort(reverse=True)
                # last_best_params = df_sort_test[0][1]
                # #### using the best parameter sets for prediction
                # if 'algorithm' in last_best_params:
                #     a.algorithm = last_best_params['algorithm']
                # a.maxlag = int(last_best_params['maxlag'])
                # a.windowsize = int(last_best_params['windowsize'])
                # a.threshold = last_best_params['threshold']
                # a.selectFeatures()
                # last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade = a.run(mode='forward')

                ## ---------------------------------------------------------------------------------------
                # reindex_list = range(num, 0, -int(num/4))+[0]
                # reindex_cross_predict_list = [reindex_list[i]-reindex_list[i+1] for i in range(len(reindex_list)-1)]
                #
                # last_best_params = last_df_sort.iloc[0].to_dict()
                # for i_reindex_cross_predict_list in range(0, len(reindex_cross_predict_list)):
                #     # a.cross_predict_days = 1
                #     a.cross_predict_periods = reindex_cross_predict_list[i_reindex_cross_predict_list]
                #     # a.cross_predict_num = num
                #     a.cross_predict_reindex = reindex_list[i_reindex_cross_predict_list+1]
                #     if 'algorithm' in last_best_params:
                #         a.algorithm = last_best_params['algorithm']
                #     a.maxlag = int(last_best_params['maxlag'])
                #     a.windowsize = int(last_best_params['windowsize'])
                #     a.threshold = last_best_params['threshold']
                #     a.selectFeatures()
                #     last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade = a.run(mode='forward')
                #     if i_reindex_cross_predict_list < len(reindex_cross_predict_list)-1:
                #         df_sort_test = []
                #         TOP = 5
                #         for i_best_params in range(TOP):
                #             last_best_params_ = last_df_sort.iloc[i_best_params].to_dict()
                #             # last_best_params_.pop('value')
                #             temp = copy.deepcopy(a)
                #             # temp.cross_predict_days = 1
                #             temp.cross_predict_periods = reindex_cross_predict_list[i_reindex_cross_predict_list]
                #             # temp.cross_predict_num = num
                #             temp.cross_predict_reindex = reindex_list[i_reindex_cross_predict_list+1]
                #             if 'algorithm' in last_best_params_:
                #                 temp.algorithm = last_best_params_['algorithm']
                #             temp.maxlag = int(last_best_params_['maxlag'])
                #             temp.windowsize = int(last_best_params_['windowsize'])
                #             temp.threshold = last_best_params_['threshold']
                #             temp.selectFeatures()
                #             last_accuracy, last_sigsum, last_adjusted_sigsum, last_trade_count, last_sig_per_trade = temp.run(mode='forward')
                #             '''you can change target here'''
                #             if temp.target == 'accuracy':
                #                 target = last_accuracy
                #             elif temp.target == 'sigsum':
                #                 target = last_sigsum
                #             elif temp.target == 'sig_per_trade':
                #                 target = last_sig_per_trade
                #             elif temp.target == 'adjusted_sigsum':
                #                 target = last_adjusted_sigsum
                #             else:
                #                 target = last_sigsum # default
                #             df_sort_test.append((target, last_best_params_))
                #             del temp
                #             gc.collect()
                #         df_sort_test.sort(reverse=True)
                #         last_best_params = df_sort_test[0][1]

                ## ---------------------------------------------------------------------------------------

            accuracy_list, sigsum, real_sigsum, adjusted_sigsum, trade_count, sig_per_trade = a.resultsDescribe(a.y_pred, a.y_targets, a.diff_targets)
            results_path = results_dir+'results_'+'resample'+str(resample_num)+'timeshift'+str(timeshift_num)
            if os.path.exists(results_path):
                shutil.rmtree(results_path)
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            with open(os.path.join(results_path, 'results.pkl'), 'wb') as fp:
                pickle.dump((accuracy_list, sigsum, real_sigsum, adjusted_sigsum), fp)
            print 'accuracy: %.4f, trade_count: %d, sigsum: %6.1f, real_sigsum: %6.1f, adjusted_sigsum: %6.1f' % \
                   (accuracy_list[-1], trade_count[-1], sigsum[-1], real_sigsum[-1], adjusted_sigsum[-1])
            a.pp(os.path.join(results_path, 'accuracy_sigsum_tradecount%d.png' % trade_count[-1]), accuracy_list, sigsum, real_sigsum, adjusted_sigsum)
            if a.logs:
                with open(results_dir+'results_pred.txt', 'w') as f_logs:
                    for str in a.results_pred:
                        f_logs.write('%s\n' % str)

    ##########################################################################################
    end_time = datetime.datetime.now()
    print 'time:', end_time-start_time


'''
先运行online=False的情况，生成参数保存，结果命名为results_dir_offline
再将生成参数部分修改为读取参数部分以保证参数一致，并改online=True，结果命名为results_dir_online
最后运行online_compare.py
'''