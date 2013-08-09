import sys
sys.path.insert(0, '..')

from serialization import save_results
from maltlog_data import get_maltlog_data,create_matrices
from pycofi.learning import learn_features
num_features = 4
lambda_val = 0.06

(items, users, by_item, by_user) = get_maltlog_data()
(Y, R, item_mean) = create_matrices(items, users, by_item, by_user)
(X, Theta, J_train, J_cross) = learn_features(Y, R, None, num_features, lambda_val, 1500)
users = dict(users)

print J_train
save_results(items, users, J_train, lambda_val, Theta, X, item_mean)
