import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

MODEL_FILE = "volume_model.pkl"


class VolumeAnomalyDetector:
    def __init__(self, window_size=5):
        """
        Initializes the Anomaly Detector.
        :param window_size: Number of past batches to consider for context.
        """
        self.window_size = window_size
        # Isolation Forest Configuration
        # contamination=0.05 means we estimate roughly 5% of data might be anomalous
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
        )
        self.is_trained = False

    def _extract_features(self, history_df):
        """
        Converts raw history (log counts) into ML features for the model.

        Features created per cluster:
        1. Current Volume
        2. Velocity (Change from previous batch)
        3. Rolling Average (Context)
        4. Deviation (How far from average?)
        """
        features = []
        cluster_ids = []

        if history_df.empty:
            return np.array([]), []

        # Group by cluster to process each cluster's specific timeline
        grouped = history_df.groupby("cluster_id")

        for cid, group in grouped:
            # We need at least 2 data points to calculate velocity
            if len(group) < self.window_size:
                continue

            # Sort by time just in case
            group = group.sort_values("batch_timestamp")
            counts = group["log_count"].values

            # 1. Current Volume (The latest entry)
            current_vol = counts[-1]

            # 2. Velocity (Current - Previous)
            prev_vol = counts[-2]
            velocity = current_vol - prev_vol

            # 3. Rolling Average (Mean of the visible window)
            rolling_avg = np.mean(counts)

            # 4. Deviation (Z-Score approximation)
            # Add small epsilon (1e-5) to prevent division by zero if std_dev is 0
            std_dev = np.std(counts) + 1e-5
            deviation = (current_vol - rolling_avg) / std_dev

            features.append([current_vol, velocity, rolling_avg, deviation])
            cluster_ids.append(cid)

        return np.array(features), cluster_ids

    def train(self, historical_data_df):
        """
        Trains the Isolation Forest on simulated or real historical data.
        """
        print("Extracting features for Volume Model training...")
        X, _ = self._extract_features(historical_data_df)

        if len(X) < 10:
            print(
                f"⚠️ Not enough data to train Volume Model (Got {len(X)} samples, need ~10+). Skipping."
            )
            return

        print(f"Training Volume Model on {len(X)} samples...")
        self.model.fit(X)
        self.is_trained = True
        print("✅ Volume Model Trained successfully.")

    def detect_anomalies(self, history_df, max_anomalies=3):
        """
        Predicts anomalies based on the latest history using score-based
        relative ranking instead of the raw binary prediction.

        Returns: List of cluster_ids that are anomalous (at most max_anomalies).
        """
        if not self.is_trained:
            print("⚠️ Volume Model is not trained. Skipping inference.")
            return []

        # 1. Check Data Sufficiency
        if history_df.empty:
            return []

        # 2. Extract Features (per-cluster filtering happens inside _extract_features)
        X, cluster_ids = self._extract_features(history_df)

        total_clusters = history_df["cluster_id"].nunique()
        print(f"Feature extraction: {len(cluster_ids)}/{total_clusters} clusters have sufficient history (window={self.window_size})")

        if len(X) == 0:
            return []

        # 3. Get anomaly SCORES instead of binary predictions.
        #    score_samples() returns negative scores; lower = more anomalous.
        scores = self.model.score_samples(X)

        # 4. Log all scores for debugging
        print("--- Volume Anomaly Scores (lower = more anomalous) ---")
        for i, cid in enumerate(cluster_ids):
            print(f"  Cluster {cid}: score={scores[i]:.4f}")

        # 5. Use relative ranking: flag clusters whose score is significantly
        #    below the batch mean (z-score < -1.0).
        #    This is distribution-agnostic — it compares clusters against
        #    each other within this batch, not against the training threshold.
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        anomalies = []
        if std_score > 1e-6:
            # There is meaningful variance — use z-score to find outliers
            z_threshold = -1.0
            scored_clusters = []
            for i, cid in enumerate(cluster_ids):
                z = (scores[i] - mean_score) / std_score
                if z < z_threshold:
                    scored_clusters.append((cid, scores[i], z))

            # Sort by score ascending (most anomalous first), cap at max_anomalies
            scored_clusters.sort(key=lambda x: x[1])
            scored_clusters = scored_clusters[:max_anomalies]

            anomalies = [sc[0] for sc in scored_clusters]

            for cid, score, z in scored_clusters:
                print(f"  🚨 FLAGGED Cluster {cid}: score={score:.4f}, z={z:.2f}")
        else:
            # All scores are nearly identical — no real anomaly in this batch
            print("  All clusters have similar scores. No anomalies detected.")

        print(f"Anomaly summary: {len(anomalies)}/{len(cluster_ids)} clusters flagged (mean={mean_score:.4f}, std={std_score:.4f})")
        return anomalies

    def save(self, directory):
        """Saves the trained model to disk."""
        if not self.is_trained:
            return

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, MODEL_FILE)
        joblib.dump(self.model, path)
        print(f"Volume model saved to {path}")

    def load(self, directory):
        """Loads the trained model from disk."""
        path = os.path.join(directory, MODEL_FILE)
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True
            print(f"Volume model loaded from {path}")
        else:
            print("⚠️ No Volume model found. Inference will be skipped.")
