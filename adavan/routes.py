import os

import requests


def get_route(origin, destination, waypoints, departure_time='now'):
    """Given a starting points, ending point, and intermediate stops, will return the optimal route for visiting
    all of the stops.

    Parameters:
        origin (str): The starting point for the trip, as a Google APIs place_id.
        destination (str): The ending point for the trip, as a Google APIs place_id.
        waypoints (list): A list of intermediate stops for the trip.
                          List elements should be given, as a Google APIs place_id.
        departure_time (str): Time of departure to be used in route calculation. Defaults to the current time.

    Returns:
        TBD
    """
    origin = 'place_id:' + origin
    destination = 'place_id:' + destination
    waypoints = ['place_id:' + point for point in waypoints]

    params = {
        'origin': origin,
        'destination': destination,
        'waypoints': '|'.join(waypoints),
        'departure_time': departure_time,
        'key': os.environ['KEY']
    }

    response = requests.get('https://maps.googleapis.com/maps/api/directions/json?', params=params).json()

    return response
