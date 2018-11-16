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
        if response['status'] != 'OK':
            return None
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
            if response['status'] != 'OK':
                return None
            place_id = response['results'][0]['place_id']
            latitude = response['results'][0]['geometry']['location']['lat']
            longitude = response['results'][0]['geometry']['location']['lng']
            converted_places.append((place_id, latitude, longitude))
        return converted_places
    else:
        raise ValueError('Positional argument \'place\' must either be a string or list of strings.')
