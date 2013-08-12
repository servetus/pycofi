from numpy import matrix, zeros, concatenate, ndarray, asarray
from random import random, shuffle

def create_fake_matrices(num_users = 20, num_beers = 100):
    num_features = 4

    features = [1,1,0,0]
    has_feature = [1,-1,0,0]

    Y = matrix( zeros( ( num_users, num_beers ) ) )
    R = matrix( zeros( ( num_users, num_beers ) ) )

    X = matrix( zeros( ( num_users, num_features ) ) )
    Theta = matrix( zeros( ( num_features, num_beers ) ) )

    for ui in range(num_users):
       shuffle( features ) 
       X[ui,:] = features

    for bi in range(num_beers):
       shuffle( has_feature ) 
       Theta[:,bi] = matrix(has_feature).T

    Y = X * Theta
    
    for ui in range(num_users):
        for bi in range(num_beers):
            R[ui,bi] = random() > 0.70 
    
    return (Y, R, X, Theta, num_features)

