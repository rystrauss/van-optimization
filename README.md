[1]: https://en.wikipedia.org/wiki/Travelling_salesman_problem
[2]: https://github.com/pypa/pipenv
[3]: https://developers.google.com/maps/documentation/directions/start#get-a-key

# Ada Jenkins Van Route Optimization

This repository contains a program for use at the Ada Jenkins Center in Davidson, NC. The Ada Jenkins Center's
LEARNWorks program partners with families, schools, and volunteers to support the academic progress of students and to
advance family engagement. The program’s students have had an increased need for transportation home. There is one van,
which makes multiple trips. This project aims to find the optimal route for the van in order to get everyone
home as quickly as possible.

We accomplish this by first performing a constrained clustering on the drop off locations to split them up into
multiple trips. For each trip, we then use the Google Directions API to solve the [TSP][1] and get the optimal route
for the van to take.

## Prerequisites

A virtual environment with the necessary prerequisites can be setup from the `Pipfile` using [pipenv][2].

## Usage

The script `calculate_van_routes.py`  provides the following command-line interface for calculating the optimal routes:

```text
Usage: calculate_van_routes.py [OPTIONS] INPUT OUTPUT

  Finds the optimal van routes for dropping off students from the Ada
  Jenkins center.

Options:
  --capacity INTEGER        The maximum number of students that can fit in the
                            van. Defaults to 13.
  --min_passengers INTEGER  The minimum number of passengers in a trip.
                            Defaults to 0.
  --extra_trips INTEGER     The maximum number of extra trips to be allowed.
                            If a clustering cannot be found with the minimum
                            number of trips possible, the program will attempt
                            to find a clustering using an additional cluster.
                            This option specifies how many times that is
                            allowed to happen. Defaults to 1.
  --origin TEXT             The starting/ending point for the trips. Given as
                            an address; use underscores where spaces would
                            normally be used (i.e.
                            "111_ABC_Street,_Davidson,_NC_28036"). Defaults to
                            the Ada Jenkins Center.
  --key TEXT                Optionally provide an API key. If not provided,
                            the program will default to the environment
                            variable "KEY".
  --help                    Show this message and exit.
```

### Input

The argument `INPUT` is a path to an input file that contains information regarding the drop off locations. This
file should have the format

```text
ADDRESS
NUM_STUDENTS
WEEKDAYS
PRIORITY

ADDRESS
NUM_STUDENTS
WEEKDAYS
PRIORITY

...
```

where `ADDRESS` is the address of the drop off location, `NUM_STUDENTS` is the number of students that live there,
`WEEKDAYS` is the days of the week that this locations is visited, and `PRIORITY` is a number representing the priority
that location should be given (0 is the lowest priority).

Example:
```text
111 ABC Street, Davidson, NC 28036
3
M T W R
1
```

### Output

The argument `OUTPUT` specifies the filename the save the results to. Results will be saved in HTML format, which can
then be opened in the browser.

### Google API Key

An important thing to note is that this program relies on a working Google API Key that is authorized to use the
_Directions API_ and the _Geocoding API_. You can get an API key for free [here][3].
