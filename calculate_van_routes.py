import math

import click
import numpy as np

from adavan.ckmeans.weighted_kmeans import cluster
from adavan.locations import convert_place
from adavan.routes import get_route

ADA_JENKINS = '212_Gamble_St,_Davidson,_NC_28036'


def read_input(input):
    """Reads an input file and stores data in a dictionary.

    Args:
        input (file): A file that hold the data to be read. This function
        assumes that the file is already open.

    Returns:
        locations (dict): A dictionary where the keys are days of the week and values
        are lists of location tuples.
    """
    data = list(filter(None, input.read().splitlines()))

    locations = {}
    for i in range(0, len(data), 4):
        address, people, days, prioritize = data[i: i + 4]
        place_id, lat, long = convert_place(address)
        for day in days.split():
            if day not in locations:
                locations[day] = []
            locations[day].append((place_id, lat, long, address, int(people), bool(int(prioritize))))

    return locations


@click.command()
@click.argument('input', type=click.File('r'), nargs=1)
@click.argument('output', type=click.File('w', lazy=True), nargs=1)
@click.option('--capacity', type=click.INT, default=13,
              help='The maximum number of students that can fit in the van. Defaults to 13.')
@click.option('--extra_trips', type=click.INT, default=1,
              help='The maximum number of extra trips to be allowed. If a clustering cannot be found with the minimum '
                   'number of trips possible, the program will attempt to find a clustering using an additional '
                   'cluster. This option specifies how many times that is allowed to happen. Defaults to 1.')
@click.option('--origin', type=click.STRING, default=ADA_JENKINS,
              help='The starting/ending point for the trips. Given as an address; use underscores where spaces would '
                   'normally be used (i.e. "111_ABC_Street,_Davidson,_NC_28036"). Defaults to the Ada Jenkins Center.')
def main(input, output, capacity, extra_trips, origin):
    # Process the input
    locations = read_input(input)
    input.close()
    del input

    # Process the origin
    origin = origin.replace('_', '')
    origin = convert_place(origin)

    waypoints = locations['M']

    # The weights to be used during clustering; corresponds to the number of people at each stop
    weights = [x[4] for x in waypoints]

    # Get the total number of passengers that need to be taken home
    total_passengers = sum(weights)

    # The minimum number of trips needed
    min_trips = math.ceil(total_passengers // capacity)

    # Get lat/long data for clustering
    clustering_data = np.array([x[1:3] for x in waypoints])

    # Cluster the locations
    cluster_assignments = None
    attempts = 0
    while not cluster_assignments and attempts <= extra_trips:
        cluster_assignments = cluster(clustering_data,
                                      k=min_trips + attempts,
                                      min_weight=0,
                                      max_weight=capacity,
                                      weights=np.array(weights),
                                      max_iter=100)
        attempts += 1

    # Based on the clustering, group waypoints into their individual trips
    trips = [[] for _ in range(max(cluster_assignments) + 1)]
    for i, place in enumerate(waypoints):
        trips[cluster_assignments[i]].append(place[0])

    # Send data to the Directions API to solve the route optimization and get back the results
    routes = [get_route(origin[0], origin[0], trip) for trip in trips]


if __name__ == '__main__':
    main()
