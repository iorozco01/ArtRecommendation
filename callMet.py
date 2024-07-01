import random
import requests
import os
import json
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# Met API URL
url = "https://collectionapi.metmuseum.org/public/collection/v1/objects"

# Create directories to save images and metadata
os.makedirs('met_images', exist_ok=True)
os.makedirs('met_metadata', exist_ok=True)


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
            print(f"Artwork with ID {object_id} is not in the public domain.")
    else:
        print(f"Artwork with ID {object_id} not found.")

    # Return None if object in location doesn't fulfill parameters
    return None


# Get image of artwork from API and download
def download_image(art_url, image_):
    img_response = requests.get(art_url)
    if img_response.status_code == 200:
        try:
            img = Image.open(BytesIO(img_response.content))
            img.save(image_path)
            print(f"Image saved to {image_path}")
        except UnidentifiedImageError:
            print(f"Failed to identify image from URL: {art_url}")
    else:
        print(f"Failed to download image from URL: {art_url}, status code: {img_response.status_code}")


response = requests.get(url)
data = response.json()
object_ids = data['objectIDs']

for art_id in object_ids:
    artwork = search_object(art_id)
    if artwork:
        # Save metadata
        metadata_path = os.path.join('met_metadata', f"{art_id}.json")
        with open(metadata_path, 'w') as m:
            json.dump(artwork, m)

        # Save image
        artwork_img_url = artwork.get('primaryImage')
        if artwork_img_url:
            image_path = os.path.join('met_images', f"{art_id}.jpg")
            download_image(artwork_img_url, image_path)
            print(f"Artwork with ID {art_id} downloaded.")
        else:
            print(f"Artwork with ID {art_id} has no primary image.")
    else:
        print(f"No artwork found for ID {art_id}")
