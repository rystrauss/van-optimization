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
        response: The Directions API response.
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

    assert response['status'] == 'OK', 'Directions API call failed.'

    return response


def parse_routes(all_routes, output):
    """Parses an API response and writes the directions to an HTML document.

    Args:
        all_routes (list): A list of the routes on each day.
        output (file): The output file to write to.

    Returns:
        None
    """
    output.write('<html><head><title>Routes</title></head><body>')
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    for i, routes in enumerate(all_routes):
        day = days[i]
        output.write("<h2>The route for " + day + "</h2>")
        routes = all_routes[i]
        for trip_num, route in enumerate(routes):
            output.write("<h3>Trip " + str(trip_num + 1) + "</h3>")
            legs = route['routes'][0]['legs']
            for leg_num in range(len(legs)):
                start = legs[leg_num]["start_address"]
                end = legs[leg_num]["end_address"]
                output.write(str(leg_num + 1) + ". from " + start + " to " + end + "<br>")
                steps = legs[leg_num]["steps"]
                for step in steps:
                    instrn = step["html_instructions"]
                    output.write(instrn + "<br>")
                output.write("<br>")
    output.write('</body>')
