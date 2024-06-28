import random
import requests
from PIL import Image
from io import BytesIO

# Met API URL
url = "https://collectionapi.metmuseum.org/public/collection/v1/objects"

# Get object with ID from Met API
def search_object(object_id):
    search_url = url + "/" + str(object_id)
    requested_art = requests.get(search_url)

    # If the requested on is valid (free-use and exists) return it
    if requested_art.status_code == 200:
        object_details = requested_art.json()
        if object_details.get('isPublicDomain', True):
            return object_details
        else:
            print("Object number " + str(object_id) + " is not in the public domain).")
    else:
        print("Object number " + str(object_id) + " not found.")

    # Retry with a random object ID if the object is not found or not in public domain
    return search_object(str(random.randint(object_ids[0], object_ids[-1])))


response = requests.get(url)
data = response.json()
object_ids = data['objectIDs']

# object_num not accurate if change is made in search_object
artwork = search_object(random.randint(object_ids[0], object_ids[-1]))
if artwork:
    artwork_img_url = artwork.get('primaryImage')
    print("Artwork details:", artwork)
    print("Primary image URL:", artwork_img_url)

# Display image
    img_response = requests.get(artwork_img_url)
    img = Image.open(BytesIO(img_response.content))
    img.show()
else:
    print("No artwork found.")
