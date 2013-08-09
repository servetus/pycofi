from numpy import matrix, asarray, apply_along_axis
from numpy.linalg import norm
import simplejson


def get_top_items(user_index, items, Theta, X, item_mean=None, R_filter=None):

    if R_filter == None:
        R_filter = set([])

    scores = X[user_index,:] * Theta

    if(item_mean != None):
        scores = matrix(item_mean).T + scores

    item_scores = zip(items, asarray(scores)[0] )

    item_scores = filter( lambda x: x[0]['id'] not in R_filter, item_scores) 

    item_scores = sorted( item_scores, key= lambda x: x[1], reverse=True )

    return item_scores

def get_similar_items(item_index, items, Theta, R_filter=None):
    
    if R_filter == None:
        R_filter = set([])

    target_item_params = Theta[:,item_index]

    diff = Theta - target_item_params

    similarity = apply_along_axis( norm, 0,diff) 
    
    item_scores = zip(items, asarray(similarity) )

    item_scores = filter( lambda x: x[0]['id'] not in R_filter, item_scores) 

    item_scores = sorted( item_scores, key= lambda x: x[1] )

    return item_scores 
