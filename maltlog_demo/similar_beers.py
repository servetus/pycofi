import sys
sys.path.insert(0, '..')

from serialization import load_results
from pycofi.queries import get_similar_items

(users, items, item_mean, J_train, Theta, X) = load_results()

for item_index, item in enumerate(items):

    beer_scores = get_similar_items(item_index, items, Theta)

    print "---", item['name'],  "---"
    print Theta[:,item_index].T

    for beer in beer_scores[1:6]:
        print beer[0]['name'], beer[1]

    print "----------------"
