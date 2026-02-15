import os
import joblib
from river import cluster

# Default file names
MODEL_FILE = "denstream_model.pkl"
PIPELINE_FILE = "river_pipeline.pkl"


def create_new_model():
    return cluster.DenStream(
        decaying_factor=0.0005,
        epsilon=0.9,
        n_samples_init=300,
    )


def save_model(model, pipeline, directory="models/production"):
    """
    Saves model and pipeline to a specific directory (Staging or Production).
    """
    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, MODEL_FILE)
    pipeline_path = os.path.join(directory, PIPELINE_FILE)

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)


def load_model(directory="models/production"):
    """
    Loads model from a specific directory.
    """
    model_path = os.path.join(directory, MODEL_FILE)
    pipeline_path = os.path.join(directory, PIPELINE_FILE)

    try:
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        print(f"Loaded model from {directory}")
        return model, pipeline
    except FileNotFoundError:
        print(f"No model found in {directory}.")
        return None, None
