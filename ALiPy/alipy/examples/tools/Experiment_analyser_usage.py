from alipy.experiment import ExperimentAnalyser

# -----------------Initialize----------------------
analyser = ExperimentAnalyser(x_axis='num_of_queries')

# -----------------Add results----------------

# Number of queries
# 2 folds, 4 queries for each fold.
radom_result = [[0.6, 0.7, 0.8, 0.9], [0.7, 0.7, 0.75, 0.85]]  
# 2 StateIO object for 2 folds experiments
analyser = ExperimentAnalyser(x_axis='num_of_queries')
analyser.add_method('random', radom_result)
# uncertainty_result = [saver1, saver2] 
# analyser.add_method('uncertainty', uncertainty_result)

# Cost sensitive
# 2 folds, 4 queries for each fold.
radom_result = [[(1, 0.6), (2, 0.7), (2, 0.8), (1, 0.9)],
                [(1, 0.7), (1, 0.7), (1.5, 0.75), (2.5, 0.85)]]  
# each State object in the saver must have the 'cost' entry.
analyser = ExperimentAnalyser(x_axis='cost')
analyser.add_method('random', radom_result)
# uncertainty_result = [saver1, saver2]  
# analyser.add_method('uncertainty', uncertainty_result)


# -----------------Plot--------------------
analyser.plot_learning_curves(title='Learning curves example', std_area=True)
