from river import compose, preprocessing, feature_extraction
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dimension = 384


def get_text_embedding(text):
    """Create a 384-dimensional text embedding."""
    embedding = embedding_model.encode(text)
    return embedding


def build_feature_dict(level, source, embedding_vector):
    """
    Merge categorical features + embedding into a single dict.
    """
    data = {"level": level, "source": source}

    for i, v in enumerate(embedding_vector):
        data[f"vec_{i}"] = v

    return data


def create_streaming_pipeline():
    """
    Create an ML pipeline with:
    - PCA dimensionality reduction
    - Standard scaling
    - OneHot encoding for level & source
    """

    vec_keys = [f"vec_{i}" for i in range(embedding_dimension)]

    numeric_pipeline = compose.Select(*vec_keys) | preprocessing.StandardScaler()

    category_pipeline = (
        compose.Select("level", "source") | preprocessing.OneHotEncoder()
    )

    pipeline = numeric_pipeline + category_pipeline
    return pipeline
