from acepy.experiment import StoppingCriteria

# ---------------Initialize----------------
stopping_criterion = StoppingCriteria(stopping_criteria='num_of_queries', value=50)
# or init by toolbox
stopping_criterion = acebox.get_stopping_criterion(stopping_criteria='num_of_queries', value=50)

# ---------------Usage----------------

while not stopping_criterion.is_stop():
    	#... Query some examples and update the StateIO object
	# Use the StateIO object to update stopping_criterion object
	stopping_criterion.update_information(saver)
# The condition is met and break the loop. 
# Reset the object for another fold.
stopping_criterion.reset()