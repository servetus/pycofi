import sys
sys.path.insert(0, '..')

import unittest
from test import create_fake_matrices
from pycofi.learning import V_to_X_Theta, X_Theta_to_V, cost, cost_grad, random_init
from numpy import array, zeros, matrix, sum

class TestCostFunction(unittest.TestCase):

    def test_V_To_XTheta(self):

        num_users = 3
        num_features = 2
        num_items = 10

        V_start = array( range( (num_users * num_features) + (num_features * num_items) ) )

        X, Theta = V_to_X_Theta(V_start, num_users, num_features, num_items)

        V_end = X_Theta_to_V(X,Theta)

        self.assertTrue( not any(V_start - V_end) )


    def testGradientChecking(self, lambda_val=0):
        epsilon = 10 ** -4
        num_users = 4
        num_items = 4

        (Y, R, X, Theta, num_features) = create_fake_matrices(num_users, num_items)

        X = matrix(zeros((num_users,num_features)))
        Theta = matrix(zeros((num_features, num_items)))

        random_init(X)
        random_init(Theta)

        V = X_Theta_to_V(X,Theta)

        V_grad = cost_grad(V, Y, R, num_users, num_features, num_items, lambda_val)

        V_grad_check = zeros( len(V_grad) ) 

        for i in range(len(V_grad_check)):
            
            grad_low = V.copy()
            grad_low[i] -= epsilon
            grad_high = V.copy()
            grad_high[i] += epsilon
            
            low = cost(grad_low, Y, R, num_users, num_features, num_items, lambda_val)
            high = cost(grad_high, Y, R, num_users, num_features, num_items, lambda_val)
            
            V_grad_check[i] = (high - low) / (epsilon * 2)
        
        diff = matrix(V_grad) - matrix(V_grad_check)
        total_error = sum( diff )

        print V_grad
        print V_grad_check
        print diff
        print total_error
        print len(V_grad)

        self.assertTrue( len(V_grad) == len(V) )
        self.assertTrue( type(V_grad) == type(V) )
        self.assertTrue( total_error < (10 ** -10) )

    def testGradientCheckingLambda(self):
        self.testGradientChecking(lambda_val = 0.3)

if __name__ == '__main__':
    unittest.main()
