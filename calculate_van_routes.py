from adavan.ckmeans.weighted_kmeans import cluster
from adavan.locations import convert_place
from adavan.routes import get_route

import numpy as np

if __name__ == '__main__':
    origin = 'Davidson College Davidson NC'
    destination = 'Davidson College Davidson NC'
    waypoints = ['21214 Legion St Cornelius, NC', '19710 S Ferry St Cornelius, NC',
                 '100 N Harbor Place Dr Unit H Davidson, NC', '750 Jetton St Davidson, NC',
                 '20609 Torrence Chapel Rd Cornelius, NC']

    origin = convert_place(origin)
    destination = convert_place(destination)
    waypoints = convert_place(waypoints)

    clustering_data = np.array([x[1:] for x in waypoints])

    cluster_assignments = cluster(clustering_data, k=2, min_weight=0, max_weight=5, max_iter=100)

    trips = [[] for _ in range(max(cluster_assignments) + 1)]

    for i, place in enumerate(waypoints):
        trips[cluster_assignments[i]].append(place[0])

    routes = [get_route(origin[0], destination[0], trip) for trip in trips]
