optforwardtest system
=====================


command:  

    python optforwardtest.py -i ./data/IF.csv -of ./results/results.csv -op ./results/results.png -nj 3 -lp 10 -dn 1 -cl knn
help:  
    
    -i the input file
    -of the output file which records time index, predict label, and close
    -op the output file which describes sigsum and accuracy
    -nj number of jobs to run in parallel
    -lp length of periods to predict, not number of points to predict
    -dn length of periods to shift
    -cl classifier
	
**'''1. DATA PREPARING'''**  
key point:  

    rs_num and nrows:
    	you can change rs_num and nrows to determine the beginning and last time for the csv file reading 
    resample_time:
    	you can change resample time, by time-resampling way or point-resampling way
    zero_propotion:
    	you can change the variable, which describes the propotion of zero, to calculate the train data label
		
**'''2. FEATURE EXTRACTION'''**  
key point:
 
	features: 
	window_size:
		length of data default, you can change it if you like
	test_size:
		diff_n default, or greater than diff_n
		
**'''3. CLASSIFICATION'''**  
* in this stage, you can achivement classification by multi_feature way or multi_classification way  
* in multi_feature way, different kinds of features may be gernerated together for one classifier  
* in multi_classification way, each classifier may output results, and you can combine them by vote or other  
* a simple classifier named mean classifier is added, based on the assumption that next close is the mean close of current window including current point

>
窗口长度是window_size，这里的diff_n即timeshift  
diff_n的取值可以是1,2,3,4,5等等，也就是说如果diff_n等于2的时候，在训练数据定义label的时候，是指当前close与其后第2个点close的对比做出的一个方向。  
换句话说，当前的close的label利用了未来第2个点的close信息。  
所以我们在划分trainset和testset的时候，test_size必须大于等于diff_n才行，才能保证trainset不会偷看testset之后的close信息。（由此得出test_size=diff_n）  
而我们要预测的是testset的label，特别是test[-1]。这就意味着，test[-1]得到的label，我们可以知道其后第2个点的close的升降。  
所以窗口滑动的时候，要保证每次都取到当前预测点其后第2个点。（由此得出step=diff_n）  
这样写入文件的，是每隔2分钟的预测点的close，index和预测的label。我们可以根据这些信息画出sigsum曲线来。  
>
而resampletime是我们每次读取窗口内数据的采样，对于最后一个点test[-1]来说，意味着前面数据的稀释。我们采样的目的是，数据可能冗余太多，所以要按照resampletime采样。  
但是对于采样后的窗口来说，里面数据的label定义还是基于diff_n来做的，也就是一个点的label是当前close与其后第2个点的close的对比做出的一个方向。  


**前推的思路：**

    先截取原始数据，窗口步长为diff_n，因为预测的是后diff_n的方向。
    再对截取的片断采样。
    注意计算label时候采用periods=int(diff_n/resample_time)，并生成特征。
    为了保证训练集相邻点之间的特征计算，resample_time应该与diff_n一致。
    测试集为int(diff_n/resample_time)。

**不前推的思路：** 

    先采样原始数据。
    注意计算label时候采用periods=int(diff_n/resample_time)，并生成特征。
    为了保证训练集相邻点之间的特征计算，resample_time应该与diff_n一致。
    再截取生成特征，窗口步长为int(diff_n/resample_time)，因为预测的是后diff_n的方向。
    测试集为int(diff_n/resample_time)。


## The Default Project for LiNing

##### Please Do Not make any change without permission~

### 一般修改的地方：

    file_path
    size1
    size2
    resample_num_list
    timeshift_num_list
    results_dir
	default_nrows
    whether timeshift equals resample or not
    num
    dynamic
    
    step
    target
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
    optunity.maximize or optunity.maximize_structured

### Optunity Revision Note

	一直遇到的一个问题就是：
	op开启了比如30个进程，其实真正在计算的并没有这么多，很多同名进程都在挂载等待。
	
	真正的原因还在于optunity内部：
	在PS0源码中suggest_from_box定义，有
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
	
	在PS0源码中optimize定义，有
	pop = [self.generate() for _ in range(self.num_particles)]
	best = None
	
	for g in range(self.num_generations):
		fitnesses = pmap(evaluate, list(map(self.particle2dict, pop)))
	
	由此，可以看到，其实真正运行的进程数，不与num_evals相关，也不与你开启的进程数相关，而是与num_particles相关：它是对每个粒子的分支进行map操作！！！！
	num_evals为3，同一时间调优就是3个，也就占3个进程，所以开30个进程的话，会出现27个等待进程
	num_evals为30，同一时间调优就是10个，也就占10个进程，所以开30个进程的话，会出现20个等待进程
	num_evals为300，同一时间调优就是20个，也就占20个进程，所以开30个进程的话，会出现10个等待进程
	但是总共寻优的次数是与num_evals相关的，可能会略小于num_evals。
