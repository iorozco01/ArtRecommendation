import os
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import json
import numpy as np
from imageTraining import *


def compute_metadata_similarity(artwork1, artwork2):
    # This is a simple example. Adjust based on your metadata structure
    style_match = artwork1['metadata'].get('style') == artwork2['metadata'].get('style')
    period_match = artwork1['metadata'].get('period') == artwork2['metadata'].get('period')
    return (style_match + period_match) / 2

def find_similar_artworks_multi(liked_artwork_ids, all_artworks, top_n=10, visual_weight=0.7):
    liked_artworks = [artwork for artwork in all_artworks if artwork['id'] in liked_artwork_ids]
    liked_features = np.mean([artwork['features'] for artwork in liked_artworks], axis=0)
    
    similarities = []
    for artwork in all_artworks:
        if artwork['id'] not in liked_artwork_ids:
            visual_sim = np.dot(liked_features, artwork['features']) / (np.linalg.norm(liked_features) * np.linalg.norm(artwork['features']))
            metadata_sim = np.mean([compute_metadata_similarity(liked, artwork) for liked in liked_artworks])
            combined_sim = visual_weight * visual_sim + (1 - visual_weight) * metadata_sim
            similarities.append((artwork['id'], combined_sim))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]