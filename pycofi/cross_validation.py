from numpy import matrix, zeros, concatenate, where
from random import shuffle

def split_R_helper(R, R_idx):
    for ui,bi in R_idx:
        R[ui,bi] = 1
    return R

def split_R(R):
    R_train = matrix(zeros(R.shape))
    R_cross = matrix(zeros(R.shape))
    R_test = matrix(zeros(R.shape))

    R_idx = [i.tolist()[0] for i in concatenate(where(R)).T]
    shuffle(R_idx)

    cross_start = int(len(R_idx) * 0.6)
    test_start = int(len(R_idx) * 0.8)

    R_train_idx = R_idx[0:cross_start - 1]
    R_cross_idx = R_idx[cross_start:test_start - 1]
    R_test_idx = R_idx[test_start: -1]

    R_train = split_R_helper(R_train, R_train_idx)
    R_cross = split_R_helper(R_cross, R_cross_idx)
    R_test = split_R_helper(R_test, R_test_idx)

    return (R_train, R_cross, R_test)

def run_params():
    (items, users, by_item, by_user) = get_maltlog_data()
    (Y, R, item_mean) = create_matrices(items, users, by_item, by_user)
    #(Y, R, _, _, _) = create_fake_matrices()
    (R_train, R_cross, R_test) = split_R(R)

    lambda_val_candidates = [0.01, 0.03, 0.1, 0.3]
    num_features_candidates = [4]

    best_J_cross = 10 ** 5
    best_num_features = 0
    best_lambda_val = 0
    best_X = None
    best_Theta = None

    for num_features in num_features_candidates:
        for lambda_val in lambda_val_candidates:
            print "Trying lambda_val:%f num_features:%d" % (lambda_val, num_features)
            (X, Theta, J_train, J_cross) = cofi(Y, R_train, R_cross, num_features, lambda_val, 1000)
            
            print "best_J_cross:%f,J_train:%f, J_cross:%f" % (best_J_cross, J_train, J_cross)
            print  "**************************************"

            if( J_cross < best_J_cross ):
                best_J_cross = J_cross
                best_num_features = num_features 
                best_lambda_val = lambda_val
                best_X = X
                best_Theta = Theta
    
    print  "********************************************************************************"
    print  "sum(R_train)", sum(R_train)
    print  "sum(R_cross)", sum(R_cross)
    print  "best_J_cross:", best_J_cross   
    print  "best_num_features:", best_num_features   
    print  "best_lambda_val:", best_lambda_val   
