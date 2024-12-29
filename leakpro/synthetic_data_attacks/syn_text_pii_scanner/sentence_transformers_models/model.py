"""Module containing Sentence Transformer models."""
from sentence_transformers import SentenceTransformer

#Set model_sen_trans
model_sen_trans = SentenceTransformer(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    tokenizer_kwargs={"clean_up_tokenization_spaces": True}
)
