# acepy: ACtive lEarning toolbox for PYthon

Authors: Ying-Peng Tang, Guo-Xiang Li, [Sheng-Jun Huang](http://parnec.nuaa.edu.cn/huangsj)

## Introduction

Acepy is a python package for experimenting with different active learning settings and algorithms. It aims to support experiment implementation with miscellaneous tool functions. These tools are designed in a low coupling way in order to let users to program the experiment project at their own customs.

Features of acepy include:

* Model independent
	- There is no limitation of the model. You may use SVM in sklearn or deep model in tensorflow as you need.
	
* Module independent
	- There is no framework limitation in our toolbox, using any tools are independent and optional.
	
* Implement your own algorithm without inheriting anything
	- There are few limitations of the user-defined functions, such as the parameters or names.
	
* Variant Settings supported
	- Noisy oracles, Multi-label, Cost effective, Feature querying, etc.
	
* Powerful tools
	- Save intermediate results of each iteration AND recover the program from any breakpoints.
	- Parallel the k-folds experiment.
	- Gathering, process and visualize the experiment results.

For more detailed introduction and tutorial, please refer to the [website of acepy]() (Coming soon).

## Setup

You can get acepy simply by:

```
pip install acepy
```

Or clone acepy source code to your local directory and build from source:

```
cd Acepy
python setup.py install
```

The dependencies of acepy are:
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
cvxpy
```

## Tools in acepy

The tool classes provided by acepy cover as many components in active learning as possible. It aims to support experiment implementation with miscellaneous tool functions. These tools are designed in a low coupling way in order to let users to program the experiment project at their own customs.

* Using `acepy.data_manipulate` to preprocess and split your data sets for experiments.

* Using `acepy.query_strategy` to invoke traditional and state-of-the-art methods.

* Using `acepy.index.IndexCollection` to manage your labeled indexes and unlabeled indexes.

* Using `acepy.metric` to calculate your model performances.

* Using `acepy.experiment.state` and `acepy.experiment.state_io` to save the intermediate results after each query and recover the program from the breakpoints.

* Using `acepy.experiment.stopping_criteria` to get some example stopping criteria.

* Using `acepy.experiment.experiment_analyser` to gathering, process and visualize your experiment results.

* Using `acepy.oracle` to implement clean, noisy, cost-sensitive oracles.

* Using `acepy.utils.multi_thread` to parallel your k-fold experiment.

### Why independent tools?

In active learning experiment, the settings in active learning are plentiful. It is very hard to write a unified class to consider every special setting. Besides, the implementation way can also be plentiful. Different users have different customs in programming.

In order to adapt various users, acepy provides multifarious independent tool classes corresponding to each module in the unified framework of active learning. In this way, the code between different parts can be implemented without limitation. Also, each independent module can be replaced by users' own implementation (without inheriting). Because the modules in acepy will not influence each other and thus can be substituted freely.

### The implemented query strategies

Acepy provide several commonly used strategies for now, and new algorithms will continue to be added in subsequent updates.

+ Informative: 
	1. Uncertainty (support ['least_confident', 'margin', 'entropy', 'distance_to_boundary'])
	2. Query_By_Committee (support ['vote_entropy', 'KL_divergence'], using a bagging method to construct committee by default.)
	3. Expected_Error_Reduction

+ Representative:
	1. Graph_Density (CVPR 2012 RALF: A reinforced active learning formulation for object class recognition)
	2. Random

+ Informative and Representative:
	1. QUIRE (TPAMI 2014 Active learning by querying informative and representative examples)
	2. (In Progress) BMDR (SIGKDD 2013 Querying Discriminative and Representative Samples for Batch Mode Active Learning)
	
+ Meta acitve learning methods:
	1. (In Progress) ALBL (AAAI 2015 Active Learning by Learning)
	2. (In Progress) LAL (NIPS 2017 Learning Active Learning from Data)

### Implement your own algorithm

In acepy, there is no limitation for your implementation. All you need is ensure the returned selected index is a subset of unlabeled indexes.

```
select_ind = my_query(unlab_ind, **my_parameters)
assert set(select_ind) < set(unlab_ind)
```
	
## Usage

There are 2 ways to use acepy. Acepy provides independent tools to ensure the scalability, thus it is recommended to follow the examples provided in the tutorial in acepy main page and pick the tools according to your usage to customize your experiment. In this way, on one hand, the logic of your program is absolutely clear to you and thus easy to debug. On the other hand, some parts in your active learning process can be substituted by your own implementation for special usage.

```
import copy
from sklearn.datasets import load_iris
from acepy.utils.toolbox import ToolBox

X, y = load_iris(return_X_y=True)
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# Use the default Logistic Regression classifier
model = acebox.get_default_model()

# The cost budget is 50 times querying
stopping_criterion = acebox.get_stopping_criterion('num_of_queries', 50)

# Use pre-defined strategy
QBCStrategy = acebox.get_query_strategy(strategy_name='QueryInstanceQBC')
QBC_result = []

for round in range(10):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = acebox.get_stateio(round)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = QBCStrategy.select(label_ind, unlab_ind, model=None, batch_size=1)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = acebox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = acebox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    QBC_result.append(copy.deepcopy(saver))

analyser = acebox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='QBC', method_results=QBC_result)
print(analyser)
analyser.plot_learning_curves(title='Example of AL', std_area=True)
```

However, some users may also need a high level encapsulation which is eaiser to use. Luckily, acepy also provides a class which has encapsulated various tools and implemented the main loop of active learning, namely acepy.experiment.AlExperiment. Note that, AlExperiment only support the most commonly used scenario - query all labels of an instance. You can run the experiments with only a few lines of codes by this class. All you need is to specify the various options, the query process will be run in multi-threaded.

```
from sklearn.datasets import load_iris
from acepy.experiment.al_experiment import AlExperiment

X, y = load_iris(return_X_y=True)
al = AlExperiment(X, y, stopping_criteria='num_of_queries', stopping_value=50,)
al.split_AL()
al.set_query_strategy(strategy="QueryInstanceUncertainty", measure='least_confident')
al.set_performance_metric('roc_auc_score')
al.start_query(multi_thread=True)
al.plot_learning_curve()
```

Note that, if you want to implement your own algorithm with this class, there are some constraints have to be satisfied, please see api reference for this class.
