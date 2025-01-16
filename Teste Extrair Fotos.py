from math import radians, sin, cos, sqrt, atan2, degrees, asin


def calculate_distance(lat1, lon1, lat2, lon2):
    # Earth's radius in kilometers
    r = 6371.0

    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = r * c

    return round(distance, 3)


def calculate_midpoint(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    # Calculate midpoint
    Bx = cos(lat2) * cos(dlon)
    By = cos(lat2) * sin(dlon)

    lat3 = atan2(sin(lat1) + sin(lat2), sqrt((cos(lat1) + Bx) * (cos(lat1) + Bx) + By * By))
    lon3 = lon1 + atan2(By, cos(lat1) + Bx)

    # Convert back to degrees and normalize longitude
    lat3 = degrees(lat3)
    lon3 = degrees(lon3)

    # Normalize longitude to be between -180 and 180
    lon3 = ((lon3 + 180) % 360) - 180

    return (round(lat3, 6), round(lon3, 6))


def calculate_azimuth(lat1, lon1, lat2, lon2):
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    # Calculate azimuth
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    azimuth = atan2(y, x)

    # Convert to degrees and normalize to 0-360
    azimuth = degrees(azimuth)
    azimuth = (azimuth + 360) % 360

    return round(azimuth, 2)


def calculate_new_coordinate(lat1, lon1, distance, bearing):
    bearing = radians(bearing)

    # Earth's radius in kilometers
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    bearing = radians(bearing)

    lat2 = asin(sin(lat1) * cos(distance / R) + cos(lat1) * sin(distance / R) * cos(bearing))
    lon2 = lon1 + atan2(
        sin(bearing) * sin(distance / R) * cos(lat1),
        cos(distance / R) - sin(lat1) * sin(lat2)
    )
    return (degrees(lat2), degrees(lon2),)


def main():
    lat1 = 38.7217579
    long1 = -9.4421441
    lat2 = 38.7218518
    long2 = -9.4415605
    print("Distance: ")
    print(calculate_distance(lat1, long1, lat2, long2))

    print("Middle Point: ")
    print(calculate_midpoint(lat1, long1, lat2, long2))

    print("Bearing: ")
    print(calculate_azimuth(lat1, long1, lat2, long2))

    print("New Coordinate: ")
    print(calculate_new_coordinate(lat1, long1, 0.052, 90))


if __name__ == "__main__":
    main()