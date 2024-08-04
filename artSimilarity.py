import torch
import numpy as np

art_features = {}

#Change art_collection to file containing all images
for art_piece in art_collection:
    features = extract_features(art_piece.image_path)
    art_features[art_piece.id] = {
        'features': features,
        'metadata': art_piece.metadata  # This could include artist, style, period, etc.
    }

def find_similar_artworks_multi(liked_artwork_ids, top_n=10, visual_weight=0.7):
    # Compute average feature vector of liked artworks
    liked_features = torch.stack([art_features[id]['features'] for id in liked_artwork_ids])
    avg_liked_features = torch.mean(liked_features, dim=0)
    
    similarities = {}
    for art_id, art_data in art_features.items():
        if art_id not in liked_artwork_ids:
            # Compute visual similarity
            visual_sim = torch.cosine_similarity(avg_liked_features, art_data['features']).item()
            
            # Compute metadata similarity (example with style and period)
            metadata_sim = compute_metadata_similarity(liked_artwork_ids, art_id)
            
            # Combine visual and metadata similarity
            combined_sim = visual_weight * visual_sim + (1 - visual_weight) * metadata_sim
            
            similarities[art_id] = combined_sim
    
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

def compute_metadata_similarity(liked_ids, candidate_id):
    liked_metadata = [art_features[id]['metadata'] for id in liked_ids]
    candidate_metadata = art_features[candidate_id]['metadata']
    
    # Example: Compare style and period
    style_match = sum(lm['style'] == candidate_metadata['style'] for lm in liked_metadata) / len(liked_ids)
    period_match = sum(lm['period'] == candidate_metadata['period'] for lm in liked_metadata) / len(liked_ids)
    
    return (style_match + period_match) / 2  # Simple average, can be weighted if needed