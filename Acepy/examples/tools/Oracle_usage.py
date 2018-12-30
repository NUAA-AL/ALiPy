from acepy.oracle import Oracle

indexes = [34, 56, 74]
labels = [0, 1, 0]

# ----------------Initialize----------------
# Initialize a Oracle in different ways
oracle1 = Oracle(labels=labels)
print(oracle1)

oracle2 = Oracle(labels=labels, indexes=indexes)
print(oracle2)

cost = [2, 1, 2]
oracle3 = Oracle(labels=labels, indexes=indexes, cost=cost)
print(oracle3)

feature_mat = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
oracle4 = Oracle(labels=labels, indexes=indexes, cost=cost, examples=feature_mat)
print(oracle4)

# ---------------Add knowledge----------------
oracle1.add_knowledge(labels=[1, 0], indexes=[23, 33], examples=[[4, 4, 4], [5, 5, 5]])
print(oracle1)
oracle2.add_knowledge(labels=[1, 0], indexes=[23, 33], examples=[[4, 4, 4], [5, 5, 5]])
oracle3.add_knowledge(labels=[1, 0], indexes=[23, 33], examples=[[4, 4, 4], [5, 5, 5]], cost=[2, 1])
oracle4.add_knowledge(labels=[1, 0], indexes=[23, 33], examples=[[4, 4, 4], [5, 5, 5]], cost=[2, 1])

# ---------------Query-----------------
# Query by index or example
queried_index = 34
labels, cost = oracle4.query_by_index(indexes=queried_index)
queried_index = [34, 56]
labels, cost = oracle4.query_by_index(indexes=queried_index)
print(labels, cost)

labels, cost = oracle4.query_by_example(queried_examples=[1, 1, 1])
labels, cost = oracle4.query_by_example(queried_examples=[4, 4, 4])
print('444',labels, cost)

# ------------Multi labe oracle-------------
from acepy.oracle import OracleQueryMultiLabel
mult_y = [[1, 1, 1], [0, 1, 1], [0, 1, 0]]
moracle = OracleQueryMultiLabel(labels=mult_y)

labels, cost = moracle.query_by_index((0, )) # query all labels of instance 0
labels, cost = moracle.query_by_index((0, 1)) # query the 2nd label of 1st instance

# ------------Multi oracles-------------
from acepy.oracle import Oracles

oracles = Oracles()
oracles.add_oracle(oracle_name='Tom', oracle_object=oracle3)
oracles.add_oracle(oracle_name='Amy', oracle_object=oracle4)

oracles.query_from(34, 'Amy')
oracle1_temp = oracles.get_oracle('Amy')
labels, cost = oracle1_temp.query_by_index(34)
print(labels, cost)
print(oracles.full_history())
