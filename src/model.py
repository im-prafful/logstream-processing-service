import joblib
from river import cluster

model_path = "models/denstream_model.pkl"
pipeline_path = "models/river_pipeline.pkl"


def create_new_model():
    return cluster.DenStream(decaying_factor=0.01, epsilon=0.1, n_samples_init=1000)


def save_model(model, pipeline):
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print(f"Saving pipeline to {pipeline_path}...")
    joblib.dump(pipeline, pipeline_path)
    print("Save complete.")


def load_model():
    try:
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        print("Loaded existing model and pipeline from disk.")
        return model, pipeline
    except FileNotFoundError:
        print("No saved models found. Creating new ones.")
        return create_new_model(), None
