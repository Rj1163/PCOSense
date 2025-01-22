import googlemaps

def find_nearest_gynecologist(latitude, longitude):
    # Initialize Google Maps client with your API key
    gmaps = googlemaps.Client(key='YOUR_API_KEY')

    # Perform a nearby search for gynecologists
    gynecologists = gmaps.places_nearby(location=(latitude, longitude), radius=10000, type='gynecologist')

    if gynecologists['status'] == 'OK':
        nearest_gynecologist = gynecologists['results'][0]
        return nearest_gynecologist['name'], nearest_gynecologist['vicinity']
    else:
        return "Error", "No gynecologist found nearby."

if __name__ == "__main__":
    # Example usage
    latitude = 37.7749  # Example latitude
    longitude = -122.4194  # Example longitude
    result = find_nearest_gynecologist(latitude, longitude)
    print(result)
