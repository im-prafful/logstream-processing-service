import os
import joblib
import numpy as np
from scipy.spatial import distance


class SemanticVectorEngine:
    def __init__(self, minkowski_p=1.5, threshold=0.35):
        """
        :param minkowski_p: 1.5 is a robust balance between Manhattan (1) and Euclidean (2).
        :param threshold: Distance threshold. Lower = stricter grouping.
        """
        self.minkowski_p = minkowski_p
        self.threshold = threshold
        # Dictionary to hold active semantic centroids: { 'semantic_id': vector_array }
        self.active_centroids = {}

    def calculate_distance(self, vec_a, vec_b):
        return distance.minkowski(vec_a, vec_b, p=self.minkowski_p)

    def get_semantic_group(self, new_vector, log_id):
        """
        Finds the closest semantic group for a new vector.
        """
        best_match_id = None
        min_dist = float("inf")

        # 1. Compare against known semantic centers
        for sem_id, centroid in self.active_centroids.items():
            dist = self.calculate_distance(new_vector, centroid)

            if dist < self.threshold and dist < min_dist:
                min_dist = dist
                best_match_id = sem_id

        # 2. Decision Logic
        if best_match_id:
            # OPTIONAL: Weighted average to slowly drift the centroid (Evolution)
            # self.active_centroids[best_match_id] = 0.9 * self.active_centroids[best_match_id] + 0.1 * new_vector
            return best_match_id
        else:
            # Create new group
            new_id = f"sem_grp_{log_id}"
            self.active_centroids[new_id] = new_vector
            return new_id

    def save(self, filepath="models/vector_centroids.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(
            f"Saving {len(self.active_centroids)} semantic centroids to {filepath}..."
        )
        joblib.dump(self.active_centroids, filepath)

    def load(self, filepath="models/vector_centroids.pkl"):
        if os.path.exists(filepath):
            self.active_centroids = joblib.load(filepath)
            print(
                f"Loaded {len(self.active_centroids)} semantic centroids from {filepath}."
            )
        else:
            print("No existing vector centroids found. Starting fresh.")
