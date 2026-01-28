import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# Filename for saving the volume model
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
            if len(group) < 2:
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

    def detect_anomalies(self, history_df):
        """
        Predicts anomalies based on the latest history.
        Returns: List of cluster_ids that are anomalous.
        """
        if not self.is_trained:
            print("⚠️ Volume Model is not trained. Skipping inference.")
            return []

        # 1. Check Data Sufficiency (The "Warm Up" Check)
        # We need to ensure we have enough history to make a valid prediction
        if history_df.empty:
            return []

        # Check max depth of history (e.g., do we have 5 batches yet?)
        max_history_depth = history_df.groupby("cluster_id").size().max()
        if max_history_depth < self.window_size:
            print(
                f"⏳ System Warming Up... (Current Depth: {max_history_depth}/{self.window_size})"
            )
            return []

        # 2. Extract Features
        X, cluster_ids = self._extract_features(history_df)

        if len(X) == 0:
            return []

        # 3. Predict (-1 = Anomaly, 1 = Normal)
        predictions = self.model.predict(X)

        anomalies = []
        for i, pred in enumerate(predictions):
            if pred == -1:
                anomalies.append(cluster_ids[i])

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
