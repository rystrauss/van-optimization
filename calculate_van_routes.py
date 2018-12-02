import math
import os

import click
import numpy as np

from adavan.ckmeans.weighted_kmeans import cluster
from adavan.locations import convert_place
from adavan.routes import get_route, parse_routes

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
    count = 0
    matrix_mapping = {}
    locations = {}
    place_ids = []
    for i in range(0, len(data), 4):
        address, people, days, prioritize = data[i: i + 4]
        place_id, lat, long = convert_place(address)
        if place_id not in place_ids:
            place_ids.append(place_id)
        if place_id not in matrix_mapping:
            matrix_mapping[(lat, long)] = count
            count += 1
        for day in days.split():
            if day not in locations:
                locations[day] = []
            locations[day].append((place_id, lat, long, address, int(people), bool(int(prioritize))))

    return locations, matrix_mapping, place_ids


@click.command()
@click.argument('input', type=click.File('r'), nargs=1)
@click.argument('output', type=click.File('w', lazy=True), nargs=1)
@click.option('--capacity', type=click.INT, default=13,
              help='The maximum number of students that can fit in the van. Defaults to 13.')
@click.option('--min_passengers', type=click.INT, default=0,
              help='The minimum number of passengers in a trip. Defaults to 0.')
@click.option('--extra_trips', type=click.INT, default=1,
              help='The maximum number of extra trips to be allowed. If a clustering cannot be found with the minimum '
                   'number of trips possible, the program will attempt to find a clustering using an additional '
                   'cluster. This option specifies how many times that is allowed to happen. Defaults to 1.')
@click.option('--origin', type=click.STRING, default=ADA_JENKINS,
              help='The starting/ending point for the trips. Given as an address; use underscores where spaces would '
                   'normally be used (i.e. "111_ABC_Street,_Davidson,_NC_28036"). Defaults to the Ada Jenkins Center.')
@click.option('--key', type=click.STRING, default=None,
              help='Optionally provide an API key. If not provided, the program will default to the '
                   'environment variable "KEY".')
def main(input, output, capacity, min_passengers, extra_trips, origin, key):
    """Finds the optimal van routes for dropping off students from the Ada Jenkins center."""
    if key:
        os.environ['KEY'] = key

    # Process the input
    locations, matrix_mapping, place_ids = read_input(input)
    input.close()
    del input

    # Process the origin
    origin = origin.replace('_', '')
    origin = convert_place(origin)

    all_routes = []

    for day, waypoints in locations.items():
        # The weights to be used during clustering; corresponds to the number of people at each stop
        weights = [x[4] for x in waypoints]

        # Get the total number of passengers that need to be taken home
        total_passengers = sum(weights)

        # The minimum number of trips needed
        min_trips = math.ceil(total_passengers / capacity)

        # Get lat/long data for clustering
        clustering_data = np.array([x[1:3] for x in waypoints])

        # Cluster the locations
        cluster_assignments = None
        attempts = 0
        while not cluster_assignments and attempts <= extra_trips:
            cluster_assignments = cluster(clustering_data,
                                          k=min_trips + attempts,
                                          max_weight=capacity,
                                          min_weight=min_passengers,
                                          weights=np.array(weights),
                                          max_iter=100)
            attempts += 1

        if cluster_assignments is None:
            print('No clustering could be found. Please double check the input.')
            return

        # Based on the clustering, group waypoints into their individual trips
        trips = [[] for _ in range(max(cluster_assignments) + 1)]
        for i, place in enumerate(waypoints):
            trips[cluster_assignments[i]].append(place)

        # Order the routes by priority
        trips.sort(key=lambda x: len(list(filter(lambda j: j[5], x))), reverse=True)
        for i in range(len(trips)):
            trips[i] = [x[0] for x in trips[i]]

        # Send data to the Directions API to solve the route optimization and get back the results
        routes = [get_route(origin[0], origin[0], trip) for trip in trips]
        all_routes.append(routes)

    parse_routes(all_routes, output)


if __name__ == '__main__':
    main()
