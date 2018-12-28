# data and observed index
X = [
    [1.3, 1.3, 2.6, 0.0],
    [1.4, 1.1, 3.0, 0.1],
    [2.6, 2.3, 3.0, 0.2],
    [2.7, 2.1, 1.0, 1.0],
]
y = [1, 1, 0, 0]
observed_ind = [(0, 0), (0, 1), (0, 2), (1, 1),
                (1, 3), (2, 0), (2, 1), (3, 0), (3, 3)]
mask = [
    [1, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [1, 0, 0, 1]]

# matrix completion
from acepy.query_strategy.query_features import AFASMC_mc
X_filled = AFASMC_mc(X=X, y=y, omega=observed_ind)
print(X_filled)

# complete with mask matrix
from acepy.query_strategy.query_features import AFASMC_mask_mc
X_filled = AFASMC_mask_mc(X=X, y=y, mask=mask)
print(X_filled)

# using svd completion method
from acepy.query_strategy.query_features import IterativeSVD_mc
svd_mc = IterativeSVD_mc(rank=3)
X_filled = svd_mc.impute(X=X, observed_mask=mask)
print(X_filled)
