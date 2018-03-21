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
