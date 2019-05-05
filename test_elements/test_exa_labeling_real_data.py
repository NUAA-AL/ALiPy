import numpy as np
from sklearn.linear_model import LogisticRegression
from alipy.query_strategy import QueryInstanceUncertainty
from alipy.index import IndexCollection
from alipy.oracle import MatrixRepository

# Your labeled set
X_lab = np.random.randn(100, 10)
y_lab = np.random.randint(low=0, high=2, size=100)
# The unlabeled pool, the labels of unlabeled data can be anything. The algorithm will not use them.
X_unlab = np.random.rand(100,10)
y_place_holder = np.random.randint(low=0, high=2, size=100)

# Initialize a query strategy.
unc = QueryInstanceUncertainty(X=np.vstack((X_unlab, X_lab)), y=np.hstack((y_place_holder, y_lab)))
unlab_ind = IndexCollection(np.arange(100))   # Indexes of your unlabeled set for querying
label_ind = IndexCollection(np.arange(start=100, stop=200))  # Indexes of your labeled set
labeled_repo = MatrixRepository(examples=X_lab, labels=y_lab, indexes=label_ind)   # Create a repository to store the labeled instances

# Initialize your model
model = LogisticRegression(solver='liblinear')
model.fit(X_lab, y_lab)

# Set the stopping criterion
for i in range(10):
    # Use a sklearn model to select instances.
    select_ind = unc.select(label_index=label_ind, unlabel_index=unlab_ind, model=model, batch_size=1)
    label_ind.update(select_ind)
    unlab_ind.difference_update(select_ind)

    # Label the selected instance here
    selected_instance = X_unlab[select_ind]
    # Replace this line with your own labeling code ! But not always labeling instances with 1.
    lab_of_ins = 1

    # Add the labeled example to the repo
    labeled_repo.update_query(labels=lab_of_ins, indexes=select_ind, examples=selected_instance)

    # If you are using your own model, update your model here, and pass it to unc.select()
    X_tr, y_tr, ind = labeled_repo.get_training_data()
    model.fit(X_lab, y_lab)

    # Update the label matrix of the query strategy here in just case that the algorithm may use the labels of labeled set
    unc.y[select_ind] = lab_of_ins

# Display the labeling history
# print(labeled_repo.full_history())

# import pickle
# with open('my_labeled_set.pkl', 'wb') as f:
#     pickle.dump(labeled_repo, f)
