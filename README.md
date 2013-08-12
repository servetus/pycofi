pycofi
======

An simple demonstration of collaborative filtering built for Maltlog.com. Maltlog.com is a small site some friends and co-workers use to track their favorite craft beers. Collaborative Filtering is an algorithm commonly used for recommendations systems, Netflix being the most notable.

Prerequisites
-------------

Pycofi has a few dependencies, numpy, requests, simplejson, etc. You probably already have them but you make sure by installing the requirements with pip. 

    pip install -r requirements.txt
    
or possibly

    sudo pip install -r requirements.txt
    
Maltlog demo
------------

There are three scripts in the maltlog_demo that demonstrate collaborative filtering using data from maltlog.com.

'learn_features.py' gets the preference data from Maltlog and finds and optimized set of features to predict that data. This portion of the process can take up to 5 minutes. The results of this learning process are saved to solution.json which must exist for the remaining scripts to work.

    python learn_features.py
    
'top_recommendations.py' takes the output of 'learn_features.py' and gives the top 10 recommendations for each user. For purposes of demonstation it simply prints the results to stdout. The script also outputs the 'homework' for each user. These are the beers for which the system is getting the strongest signal but not necessarily the highest recommendation.

    python top_recommendations.py
    
'similar_beers.py' gives the 5 most similar beers in the learned feature space for each beer in the system. This has a few quirks right now. Many beers have too few data points so their feature vector is very near the origin.

    python similar_beers.py

History
-------

Maltlog started as a whiteboard in an office @neonepiphany and @jordandandersen and others used to track their preference for the daily 4:45pm craft beer tasting. That whiteboard filled up quickly thus resulting in the creation of Maltlog.com. Before long we started to have a fairly large data set so we just had to start doing some analytics on it. This repo is the result.
