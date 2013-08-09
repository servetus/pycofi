from numpy import matrix, zeros
from collections import defaultdict
import requests

def get_maltlog_data():

    r = requests.get('''https://maltlog.com/api/logs.php''')

    data = r.json()

    by_user = defaultdict(list)
    by_beer = defaultdict(list)

    data = [point for point in data if point["rating_real_value"] != None]

    for point in data:
        by_user[point['user_username']].append(point)

    for point in data:
        by_beer[point['beer_id']].append(point)

    beers_ids = by_beer.keys()
    users = [(user, {"R_indices":list(set([tasting['beer_id'] for tasting in by_user[user]])),"index":index } ) for index, user in enumerate(by_user.keys())]

    beers_ids= filter( lambda x: len(by_beer[x]) > 1, beers_ids) 
    
    beers = [{"id":by_beer[x][0]["beer_id"], "name":by_beer[x][0]["beer_name"]} for x in beers_ids] 

    return (beers, users, by_beer, by_user)

def create_matrices(beers, users, by_beer, by_user):
    
    beer_mean = matrix( zeros( ( len(beers), 1 ) ) )
    Y = matrix( zeros( ( len(users), len(beers) ) ) )
    R = matrix( zeros( ( len(users), len(beers) ) ) )

    for ui, user in enumerate(users):
        for bi, beer in enumerate(beers):

            R_indices = by_beer[beer["id"]]
            R_indices = [t for t in R_indices if t["user_username"] == user[0] ]

            if(len(R_indices) > 0):
                R[ui,bi] = 1
                Y[ui,bi] = sum([float(x["rating_real_value"]) for x in R_indices])/len(R_indices) 

    for bi, beer in enumerate(beers):
        beer_mean[bi] = sum(Y[:,bi]) / sum(R[:,bi]) 

    for ui, user in enumerate(users):
        Y[ui,:] -= beer_mean.T

    return (Y, R, beer_mean)
