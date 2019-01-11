from alipy.query_strategy.cost_sensitive import select_POSS
budget = 20
costs = [1, 2, 5, 6, 7, 9]
value = [-1, -6, -18, -22, -28, -36]
max_value, select_index = select_POSS(value, costs, budget)
print(max_value)
print(select_index)
