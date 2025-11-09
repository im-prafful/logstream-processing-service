
from river import compose
from river import preprocessing
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_dimension = 384


def get_text_embedding(text):

    embedding = embedding_model.encode(text)
    return embedding


def create_streaming_pipeline():
    """
    Create a data processing pipeline for logs.
    This pipeline handles:
      - The numeric vector data (the 384 numbers from embeddings)
      - The categorical data like 'level' and 'source'
    """

    # Step 1: Make a list of all embedding feature names
    # e.g. vec_0, vec_1, ..., vec_383
    vec_keys = []
    for i in range(embedding_dimension):
        name = "vec_" + str(i)
        vec_keys.append(name)

    # Step 2: Handle numeric features (the 384 embedding values)This creates a River transformer — basically a small object that picks out only the keys you tell it to.
    numeric_pipeline =  compose.Select(*vec_keys) | preprocessing.StandardScaler()


    category_pipeline = compose.Select("level", "source")|preprocessing.OneHotEncoder(handle_unknown="ignore")

    # Step 4: This creates a parallel union — meaning both parts run on the same data but different columns.Internally, River calls this compose.TransformerUnion.
    # (the "+" symbol means "combine features" in River)
    pipeline = numeric_pipeline + category_pipeline

    #Your create_streaming_pipeline() just built and returned one of River’s pipeline objects, which inherits those methods like learnOne(), transformOne()...
    
    return pipeline #this is a river object not a traditional python object thats why we can do pipeline.learnOne() during training
