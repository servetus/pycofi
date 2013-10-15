from numpy import matrix, zeros, linalg, concatenate, where, multiply, subtract
from numpy import sum, ndarray, dot, asarray
from random import random
from scipy import optimize

def V_to_X_Theta(V, num_users, num_features, num_items):
    V = matrix(V).reshape( (-1,1) )
    n = (num_users * num_features)
    X = V[0:n].reshape( (num_users, num_features) )
    Theta = V[n:].reshape( (num_features, num_items) )

    return (X,Theta)

def X_Theta_to_V(X,Theta):
    V = concatenate( ( X.reshape( -1, 1), Theta.reshape( -1, 1) ) )
    return ndarray.flatten(asarray( V.T ) )

def random_init(m):
    """initializes a numpy.matrix with small,random values"""

    epsilon = 10 ** -4

    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            m[x,y] = (2 * random() * epsilon) - epsilon

def cost(V, Y, R, num_users, num_features, num_items, lambda_val):
    """
    The cost function for collaborative filtering optimization

    :param V: a one dimensional list containing all the paramaters for X and Theta
    :param Y: a num_users by num_items matrix representing the users rating of the item
    :param R: a num_users by num_items matrix with a 1 in for each item the user has rated, 0 otherwise
    :param num_users: number of users the system is optimizing for
    :param num_features: number of features the system is finding for each item an user
    :param num_items: number of users the system is optimizing for
    :param lambda_val: the reglarization parameter
    :returns: scalar cost of V given Y and R
    """

    X,Theta = V_to_X_Theta(V, num_users, num_features, num_items)

    p = multiply(subtract(X * Theta, Y), R) 

    J = sum( multiply( p,p) ) / 2.0    
    J += (lambda_val / 2.0) * sum( multiply( Theta, Theta))
    J += (lambda_val / 2.0) * sum( multiply( X, X))

    return J


def cost_grad(V, Y, R, num_users, num_features, num_items, lambda_val):
    """
    The gradient cost function for collaborative filtering optimization

    :param V: a one dimensional list containing all the paramaters for X and Theta
    :param Y: a num_users by num_items matrix representing the users rating of the item
    :param R: a num_users by num_items matrix with a 1 in for each item the user has rated, 0 otherwise
    :param num_users: number of users the system is optimizing for
    :param num_features: number of features the system is finding for each item an user
    :param num_items: number of users the system is optimizing for
    :param lambda_val: the reglarization parameter
    :returns: the partial derivative of cost for each parameter of V at V
    """
    X,Theta = V_to_X_Theta(V, num_users, num_features, num_items)

    p = multiply(subtract(X * Theta, Y), R) 

    X_grad = p * Theta.T
    X_grad += X * lambda_val
    
    Theta_grad = (p.T * X).T
    Theta_grad += Theta * lambda_val

    return X_Theta_to_V(X_grad, Theta_grad)

    
def learn_features(Y, R_train, num_features, lambda_val, R_cross=None, maxiter=1000):
    """
    Finds the optimal matrices X and Theta. 

    :param Y: a num_users by num_items mat Finds the optimal matrices X and Theta.rix representing the users rating of the item
    :param R_train: a num_users by num_items matrix with a 1 in for each item the user has rated, 0 otherwise, this will be used for training
    :param R_cross: a num_users by num_items matrix with a 1 in for each item the user has rated, 0 otherwise, this will be used to calculate the cross-validation cost
    :param num_features: number of features the system is finding for each item an user
    :returns: tuple of (X,Theta, J_train, J_cross) X a matrix representing users preferences for the features. Theta is a matrix representing each items demonstration of those features. J_train the final optimized cost on R_train. J_cross the final optimized cost on R_cross. 

    """
    
    num_users = Y.shape[0]
    num_items = Y.shape[1]
    
    X = matrix(zeros((num_users,num_features)))
    Theta = matrix(zeros((num_features, num_items)))

    random_init(X)
    random_init(Theta)

    V = X_Theta_to_V(X,Theta)

    f = lambda x: cost( x, Y, R_train, num_users, num_features, num_items, lambda_val)
    fprime = lambda x: cost_grad( x, Y, R_train, num_users, num_features, num_items, lambda_val)

    V_opt = optimize.fmin_bfgs(f, fprime=fprime, x0=V, maxiter=maxiter)
    
    X,Theta = V_to_X_Theta(V_opt, num_users, num_features, num_items)

    J_train = cost( V_opt, Y, R_train, num_users, num_features, num_items, 0)

    if(R_cross != None):
        J_cross = cost( V_opt, Y, R_cross, num_users, num_features, num_items, 0)
    else:
        J_cross = 0

    return (X,Theta, J_train, J_cross)

