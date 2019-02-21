# ALiPy: Active Learning in Python

Authors: Ying-Peng Tang, Guo-Xiang Li, [Sheng-Jun Huang](http://parnec.nuaa.edu.cn/huangsj)

Online document: [http://parnec.nuaa.edu.cn/huangsj/alipy/](http://parnec.nuaa.edu.cn/huangsj/alipy/)

Offline document: [http://parnec.nuaa.edu.cn/huangsj/alipy/offline_ver/alipy_doc_v101.zip](http://parnec.nuaa.edu.cn/huangsj/alipy/offline_ver/alipy_doc_v101.zip)

## Introduction

ALiPy是一个基于Python实现的主动学习工具包，内置20余种主动学习算法，并提供包括数据处理、结果可视化等工具。ALiPy根据主动学习框架的不同部件提供了若干独立的工具类，这样一方面可以方便地支持不同主动学习场景，另一方面可以使用户自由地组织自己的项目，用户可以不必继承任何接口来实现自己的算法与替换项目中的部件。此外，ALiPy不仅支持多种不同的主动学习场景，如标注代价敏感，噪声标注者，多标记查询等。详细的介绍与文档请参考工具包的[官方网站](http://parnec.nuaa.edu.cn/huangsj/alipy/)。

ALiPy provides a module based implementation of active learning framework, which allows users to conveniently evaluate, compare and analyze the performance of active learning methods. It implementations more than 20 algorithms and also supports users to easily implement their own approaches under different settings.

Features of alipy include:

* Model independent
	- There is no limitation to the classification model. One can use SVM in sklearn or deep model in tensorflow as you need.
	
* Module independent
	- One can freely modify one or more modules of the toolbox without affection to the others.
	
* Implement your own algorithm without inheriting anything
	- There are few limitations of the user-defined functions, such as the parameters or names.
	
* Variant Settings supported
	- Noisy oracles, Multi-label, Cost effective, Feature querying, etc.
	
* Powerful tools
	- Save intermediate results of each iteration AND recover the program from any breakpoints.
	- Parallel the k-folds experiment.
	- Gathering, process and visualize the experiment results.
	- Provide 25 algorithms.
	- Support 7 different settings.

For more detailed introduction and tutorial, please refer to the [website of alipy](http://parnec.nuaa.edu.cn/huangsj/alipy).

## Setup

You can get alipy simply by:

```
pip install alipy
```

Or clone alipy source code to your local directory and build from source:

```
cd ALiPy
python setup.py install
```

The dependencies of alipy are:
1. Python dependency

```
python >= 3.4
```
        
2. Basic Dependencies

```
numpy
scipy
scikit-learn
matplotlib
prettytable
```

3. Optional dependencies

```
cvxpy
```

Note that, the basic dependencies must be installed, and the optional dependencies are required only if users need to involke KDD'13 BMDR and AAAI'19 SPAL methods in alipy. (cvxpy will not be installed through pip.)

## Tools in alipy

The tool classes provided by alipy cover as many components in active learning as possible. It aims to support experiment implementation with miscellaneous tool functions. These tools are designed in a low coupling way in order to let users to program the experiment project at their own customs.

* Using `alipy.data_manipulate` to preprocess and split your data sets for experiments.

* Using `alipy.query_strategy` to invoke traditional and state-of-the-art methods.

* Using `alipy.index.IndexCollection` to manage your labeled indexes and unlabeled indexes.

* Using `alipy.metric` to calculate your model performances.

* Using `alipy.experiment.state` and `alipy.experiment.state_io` to save the intermediate results after each query and recover the program from the breakpoints.

* Using `alipy.experiment.stopping_criteria` to get some example stopping criteria.

* Using `alipy.experiment.experiment_analyser` to gathering, process and visualize your experiment results.

* Using `alipy.oracle` to implement clean, noisy, cost-sensitive oracles.

* Using `alipy.utils.multi_thread` to parallel your k-fold experiment.

### The implemented query strategies

ALiPy provide several commonly used strategies for now, and new algorithms will continue to be added in subsequent updates.

* AL with Instance Selection: Uncertainty (SIGIR 1994), Graph Density (CVPR 2012), QUIRE (TPAMI 2014), SPAL (AAAI 2019), Query By Committee (ICML 1998), Random, BMDR (KDD 2013), LAL (NIPS 2017), Expected Error Reduction (ICML 2001)

* AL for Multi-Label Data: AUDI (ICDM 2013) , QUIRE (TPAMI 2014) , Random, MMC (KDD 2009), Adaptive (IJCAI 2013)

* AL by Querying Features: AFASMC (KDD 2018) , Stability (ICDM 2013) , Random

* AL with Different Costs: HALC (IJCAI 2018) , Random , Cost performance

* AL with Noisy Oracles: CEAL (IJCAI 2017) , IEthresh (KDD 2009) , All, Random

* AL with Novel Query Types: AURO (IJCAI 2015)

* AL for Large Scale Tasks: Subsampling

### Implement your own algorithm

In alipy, there is no limitation for your implementation. All you need is ensure the returned selected index is a subset of unlabeled indexes.

```
select_ind = my_query(unlab_ind, **my_parameters)
assert set(select_ind) < set(unlab_ind)
```
	
## Usage

There are 2 ways to use alipy. For a high-level encapsulation, you can use alipy.experiment.AlExperiment class. Note that, AlExperiment only support the most commonly used scenario - query all labels of an instance. You can run the experiments with only a few lines of codes by this class. All you need is to specify the various options, the query process will be run in multi-threaded. Note that, if you want to implement your own algorithm with this class, there are some constraints have to be satisfied, please see api reference for this class.

```
from sklearn.datasets import load_iris
from alipy.experiment.al_experiment import AlExperiment

X, y = load_iris(return_X_y=True)
al = AlExperiment(X, y, stopping_criteria='num_of_queries', stopping_value=50,)
al.split_AL()
al.set_query_strategy(strategy="QueryInstanceUncertainty", measure='least_confident')
al.set_performance_metric('roc_auc_score')
al.start_query(multi_thread=True)
al.plot_learning_curve()
```

To customize your own active learning experiment, it is recommended to follow the examples provided in the ALiPy/examples and tutorial in [alipy main page](http://parnec.nuaa.edu.cn/huangsj/alipy), pick the tools according to your usage. In this way, on one hand, the logic of your program is absolutely clear to you and thus easy to debug. On the other hand, some parts in your active learning process can be substituted by your own implementation for special usage.

```
import copy
from sklearn.datasets import load_iris
from alipy import ToolBox

X, y = load_iris(return_X_y=True)
alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
alibox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# Use the default Logistic Regression classifier
model = alibox.get_default_model()

# The cost budget is 50 times querying
stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 50)

# Use pre-defined strategy
QBCStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceQBC')
QBC_result = []

for round in range(10):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = QBCStrategy.select(label_ind, unlab_ind, model=None, batch_size=1)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    QBC_result.append(copy.deepcopy(saver))

analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='QBC', method_results=QBC_result)
print(analyser)
analyser.plot_learning_curves(title='Example of AL', std_area=True)
```

## Citation

Please cite our work:

```
Tang, Y.-P.; Li, G.-X.; and Huang, S.-J. 2019. ALiPy: Active learning in python. 
Technical report, Nanjing University of Aeronautics and Astronautics. 
available as arXiv preprint https://arxiv.org/abs/1901.03802.
```

```
@techreport{TLHalipy,
  author = {Ying-Peng Tang and Guo-Xiang Li and Sheng-Jun Huang},
  title = {{ALiPy}: Active Learning in Python},
  institution = {Nanjing University of Aeronautics and Astronautics},
  url = {https://github.com/NUAA-AL/ALiPy},
  note = {available as arXiv preprint \url{https://arxiv.org/abs/1901.03802}},
  month = jan,
  year = 2019
}
```
