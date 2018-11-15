import os

import requests


def get_place_id(place):
    """Given a place or list of places, as addresses, returns the Google APIs place_id(s).

    Arguments:
        place (str, list): A place, or a list of places, as conventionally written addresses.

    Returns:
        place_id (str, list): The corresponding place_id(s).
    """
    if type(place) is str:
        params = {
            'key': os.environ['KEY'],
            'address': place
        }
        response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?', params=params).json()
        place_id = response['results'][0]['place_id']

        return place_id
    elif type(place) is list:
        converted_places = []

        for point in place:
            params = {
                'key': os.environ['KEY'],
                'address': point
            }
            response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?', params=params).json()
            place_id = response['results'][0]['place_id']
            converted_places.append(place_id)

            return converted_places
    else:
        raise ValueError('Positional argument \'place\' must either be a string or list of strings.')


def get_route(origin, destination, waypoints):
    """Given a starting points, ending point, and intermediate stops, will return the optimal route for visiting
    all of the stops.

    Parameters:
        origin (str): The starting point for the trip, as a Google APIs place_id.
        destination (str): The ending point for the trip, as a Google APIs place_id.
        waypoints (list): A list of intermediate stops for the trip.
                          List elements should be given, as a Google APIs place_id.

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
        'departure_time': 'now',
        'key': os.environ['KEY']
    }

    response = requests.get('https://maps.googleapis.com/maps/api/directions/json?', params=params).json()

    return response


def main():
    pass


if __name__ == '__main__':
    main()
