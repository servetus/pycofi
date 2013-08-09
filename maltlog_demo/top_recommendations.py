import sys
sys.path.insert(0, '..')

from serialization import load_results
from pycofi.queries import get_top_items

(users, items, item_mean, J_train, Theta, X) = load_results()

for user in users:

    R_filter = set(users[user]["R_indices"])

    top_recommendations = get_top_items(users[user]["index"], 
                                items, 
                                Theta, 
                                X, 
                                item_mean=item_mean, 
                                R_filter=R_filter
                                )

    homework = get_top_items(users[user]["index"], 
                                items, 
                                Theta, 
                                X, 
                                R_filter=R_filter
                                )

    #this would be a good point to filter out low-availablity beer

    print "---", user, "'s top recommendations---"

    for beer in top_recommendations[0:10]:
        print beer[0]['name'], beer[1]

    print "---", user, "'s homework---"

    for beer in homework[:5]:
        print beer[0]['name'], beer[1]

    for beer in homework[-5:]:
        print beer[0]['name'], beer[1]

    print "----------------"
    print ""

    
