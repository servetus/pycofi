def create_fake_matrices():
    num_users = 20 
    num_features = 4
    num_beers = 100 

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


def gradient_checking():
    (Y, R, X, Theta,num_features) = create_fake_matrices()
    (R_train, R_cross, R_test) = split_R(R)

    num_users = Y.shape[0]
    num_beers = Y.shape[1]

    random_init(X)
    random_init(Theta)

    V = concatenate( ( X.reshape( -1, 1), Theta.reshape( -1, 1) ) ) 
    V = ndarray.flatten(asarray( V.T ) )

    J = cost(V, Y, R, num_users, num_features, num_beers, 0)
    V_grad = cost_grad(V, Y, R, num_users, num_features, num_beers, 0)

    epsilon = 10 ** -4

    print V_grad.shape

    for xi in range(V_grad.shape[0]):
        V_plus_E = V
        V_plus_E[xi] += epsilon
        J_plus_E = cost(V_plus_E, Y, R, num_users, num_features, num_beers, 0)
        
        grad = (J_plus_E - J) / epsilon

        print xi, V_grad[xi], grad

def test_cost():

    (Y, R, X, Theta,num_features) = create_fake_matrices()
    (R_train, R_cross, R_test) = split_R(R)

    num_users = Y.shape[0]
    num_beers = Y.shape[1]

    V = concatenate( ( X.reshape( -1, 1), Theta.reshape( -1, 1) ) ) 
    V = ndarray.flatten(asarray( V.T ) )

    J = cost(V, Y, R_cross, num_users, num_features, num_beers, 0.0)

    print J
