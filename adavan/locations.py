import os

import requests


def convert_place(place):
    """Given a place or list of places, as addresses, returns the Google APIs place_id, latitude, and longitude.

    Arguments:
        place (str, list): A place, or a list of places, as conventionally written addresses.

    Returns:
        place_id (str, list): The corresponding place_id, latitude, and longitude, or None if a query fails.
    """
    if type(place) is str:
        params = {
            'key': os.environ['KEY'],
            'address': place
        }
        response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?', params=params).json()

        assert response['status'] == 'OK', 'Geocoding API call failed.'

        place_id = response['results'][0]['place_id']
        latitude = response['results'][0]['geometry']['location']['lat']
        longitude = response['results'][0]['geometry']['location']['lng']
        return place_id, latitude, longitude
    elif type(place) is list:
        converted_places = []
        for point in place:
            params = {
                'key': os.environ['KEY'],
                'address': point
            }
            response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?', params=params).json()

            assert response['status'] == 'OK', 'Geocoding API call failed.'

            place_id = response['results'][0]['place_id']
            latitude = response['results'][0]['geometry']['location']['lat']
            longitude = response['results'][0]['geometry']['location']['lng']
            converted_places.append((place_id, latitude, longitude))
        return converted_places
    else:
        raise ValueError('Positional argument \'place\' must either be a string or list of strings.')


def get_distance_matrix(places):
    places = ['place_id:' + x for x in places]
    params = {
        'key': os.environ['KEY'],
        'origins': '|'.join(places),
        'destinations': '|'.join(places)
    }
    response = requests.get('https://maps.googleapis.com/maps/api/distancematrix/json?', params=params).json()

    assert response['status'] == 'OK', 'The Distance Matrix API could not fetch the data.'

    n = len(places)
    distance_matrix = [[0] * n for _ in range(n)]

    for i, row in enumerate(response['rows']):
        for j, item in enumerate(row['elements']):
            distance_matrix[i][j] = item['duration']['value']

    return distance_matrix
