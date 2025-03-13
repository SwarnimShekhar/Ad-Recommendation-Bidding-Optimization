import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, processed_data_path="data/processed_data.csv"):
        self.data = pd.read_csv(processed_data_path)
        # Create a user-ad interaction matrix
        self.interaction_matrix = self.data.pivot_table(
            index="user_id", columns="ad_id", values="click", fill_value=0
        )
        # Compute similarity matrices for collaborative filtering
        self.user_similarity = cosine_similarity(self.interaction_matrix)
        self.ad_similarity = cosine_similarity(self.interaction_matrix.T)

    def recommend_collaborative(self, user_id, top_n=5):
        """User-User Collaborative Filtering Recommendation."""
        if user_id not in self.interaction_matrix.index:
            return []
        user_idx = self.interaction_matrix.index.get_loc(user_id)
        sim_scores = self.user_similarity[user_idx]
        # Exclude the user itself by zeroing its similarity score
        sim_scores[user_idx] = 0
        # Weighted sum of other users' interactions
        weighted_interactions = np.dot(sim_scores, self.interaction_matrix.values)
        ad_indices = np.argsort(weighted_interactions)[::-1]
        recommended_ads = self.interaction_matrix.columns[ad_indices][:top_n].tolist()
        return recommended_ads

    def recommend_content_based(self, ad_id, top_n=5):
        """Content-Based Filtering using ad similarity."""
        if ad_id not in self.interaction_matrix.columns:
            return []
        ad_idx = list(self.interaction_matrix.columns).index(ad_id)
        sim_scores = self.ad_similarity[ad_idx]
        sim_scores[ad_idx] = 0  # Exclude the ad itself
        ad_indices = np.argsort(sim_scores)[::-1]
        recommended_ads = self.interaction_matrix.columns[ad_indices][:top_n].tolist()
        return recommended_ads

    def recommend_hybrid(self, user_id, top_n=5):
        """Hybrid recommendation: average of collaborative and content-based."""
        collab_ads = set(self.recommend_collaborative(user_id, top_n=top_n*2))
        # For content-based part, we take the ads the user clicked on
        user_clicks = self.data[self.data["user_id"] == user_id]["ad_id"].unique()
        content_ads = set()
        for ad in user_clicks:
            content_ads.update(self.recommend_content_based(ad, top_n=top_n))
        # Combine both and rank by frequency (here simply take the union)
        combined = list(collab_ads.union(content_ads))
        return combined[:top_n]

if __name__ == "__main__":
    rec = Recommender()
    test_user = rec.interaction_matrix.index[0]
    print("Collaborative recommendation for user", test_user, ":", rec.recommend_collaborative(test_user))
    print("Content-based recommendation for ad 1:", rec.recommend_content_based(1))
    print("Hybrid recommendation for user", test_user, ":", rec.recommend_hybrid(test_user))