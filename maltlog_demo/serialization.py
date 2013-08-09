import simplejson
from numpy import asarray, matrix

def save_results(items, users, J_train, lambda_val, Theta, X, item_mean, filename='solution.json'):

    data = {}

    data["items"] = items
    data["users"] = users
    data["J_train"] = J_train
    data["lambda_val"] = lambda_val
    data["Theta"] = asarray(Theta).tolist() 
    data["X"] = asarray(X).tolist()
    data["item_mean"] = asarray(item_mean).tolist()

    text = simplejson.dumps(data, indent=4 * ' ')

    f = open(filename, 'w')
    f.write(text)
    f.close()

def load_results(filename='solution.json'):
    f = open(filename, 'r')
    data = simplejson.loads(f.read())
    f.close()
    data['X'] = matrix(data['X'])
    data['Theta'] = matrix(data['Theta'])
    return map( data.get, ('users', 'items', 'item_mean', 'J_train', 'Theta', 'X') )
