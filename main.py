# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from artSimilarity import *
from imageTraining import *

def main():
    liked_artwork_ids = ['12342', '56789', '98765']  # Replace with actual IDs
    recommendations = find_similar_artworks_multi(liked_artwork_ids, all_artworks)

    print("Recommended artworks:")
    for artwork_id, similarity in recommendations:
        print(f"Artwork ID: {artwork_id}, Similarity: {similarity:.4f}")


