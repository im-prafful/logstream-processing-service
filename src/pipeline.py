from river import compose, preprocessing
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dimension = 384


def get_text_embedding(text: str):
    return embedding_model.encode(text)


def create_streaming_pipeline():
    vec_keys = [f"vec_{i}" for i in range(embedding_dimension)]

    numeric_pipeline = compose.Select(*vec_keys) | preprocessing.StandardScaler()

    category_pipeline = compose.Select(
        "level", "source", "method"
    ) | preprocessing.OneHotEncoder(handle_unknown="ignore")

    pipeline = numeric_pipeline + category_pipeline
    return pipeline
