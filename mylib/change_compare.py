#coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

file_path1 = 'results_pred_change.txt'
data1 = pd.read_csv(file_path1, header=None, index_col=0, names=['index','label_change'], sep='\t')
# print data1
file_path2 = 'results_pred_nochange.txt'
data2 = pd.read_csv(file_path2, header=None, index_col=0, names=['index','label_nochange'], sep='\t')
# print data2
data = pd.concat([data1, data2], axis=1)
date_list = []
time_list = []
time_stage_list = []
for idx in data.index:
    date, time = idx.split(' ')
    date_list.append(date)
    time_list.append(time)
    # if '09:01:00'<=time<='11:30:00':
    #     time_stage_list.append('morning')
    # elif '13:01:00'<=time<='15:00:00':
    #     time_stage_list.append('afternoon')
    # elif '21:01:00'<=time<='23:59:59' or '00:00:00'<=time<='01:00:00':
    #     time_stage_list.append('evening')
    # else:
    #     time_stage_list.append('')
    if '09:01:00'<=time<='09:30:00':
        time_stage_list.append('09:01:00-09:30:00')
    elif '09:31:00'<=time<='10:00:00':
        time_stage_list.append('09:31:00-10:00:00')
    elif '10:01:00'<=time<='10:30:00':
        time_stage_list.append('10:01:00-10:30:00')
    elif '10:31:00'<=time<='11:00:00':
        time_stage_list.append('10:31:00-11:00:00')
    elif '11:01:00'<=time<='11:30:00':
        time_stage_list.append('11:01:00-11:30:00')
    elif '13:31:00'<=time<='14:00:00':
        time_stage_list.append('13:31:00-14:00:00')
    elif '14:01:00'<=time<='14:30:00':
        time_stage_list.append('14:01:00-14:30:00')
    elif '14:31:00'<=time<='15:00:00':
        time_stage_list.append('14:31:00-15:00:00')
    elif '21:01:00'<=time<='21:30:00':
        time_stage_list.append('21:01:00-21:30:00')
    elif '21:31:00'<=time<='22:00:00':
        time_stage_list.append('21:31:00-22:00:00')
    elif '22:01:00'<=time<='22:30:00':
        time_stage_list.append('22:01:00-22:30:00')
    elif '22:31:00'<=time<='23:00:00':
        time_stage_list.append('22:31:00-23:00:00')
    elif '23:01:00'<=time<='23:30:00':
        time_stage_list.append('23:01:00-23:30:00')
    elif '23:31:00'<=time<='23:59:59' or time=='00:00:00':
        time_stage_list.append('23:31:00-00:00:00')
    elif '00:01:00'<=time<='00:30:00':
        time_stage_list.append('00:01:00-00:30:00')
    elif '00:31:00'<=time<='01:00:00':
        time_stage_list.append('00:31:00-01:00:00')
    else:
        time_stage_list.append('')

data['date'] = date_list
data['time'] = time_list
data['time_stage'] = time_stage_list
data['bool_different'] = pd.Series(data.label_change != data.label_nochange).astype(int)
print data.head()
dif_rate = data['bool_different'].sum()*1.0/data.shape[0]
print 'different rate:%.2f%%' % (100*dif_rate)
data_dif = data.ix[data.label_change != data.label_nochange]
# print data_dif.head()
# data_dif_groupsize = data_dif.groupby(data_dif.time).size().sort_values(ascending=True)
data_dif_groupsize = data_dif.groupby(data_dif.time_stage).size().sort_values(ascending=True)
print data_dif_groupsize
plt.figure(figsize=(15,10))
data_dif_groupsize.plot.barh()
plt.title('different rate:%.2f%%' % (100*dif_rate))
plt.savefig('data_different_groupsize.png')
