from river import compose, preprocessing
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dimension = 384


def get_text_embedding(text):
    return embedding_model.encode(text)


def build_feature_dict(level, source, embedding_vector, semantic_id=None):
    """
    UPDATED: Now accepts 'semantic_id' to add as a feature.
    """
    data = {"level": level, "source": source}

    if semantic_id:
        data["semantic_group"] = semantic_id
    else:
        data["semantic_group"] = "unknown"

    for i, v in enumerate(embedding_vector):
        data[f"vec_{i}"] = v

    return data


def create_streaming_pipeline():
    vec_keys = [f"vec_{i}" for i in range(embedding_dimension)]

    numeric_pipeline = compose.Select(*vec_keys) | preprocessing.StandardScaler()

    category_pipeline = (
        compose.Select("level", "source", "semantic_group")
        | preprocessing.OneHotEncoder()
    )

    pipeline = numeric_pipeline + category_pipeline
    return pipeline
