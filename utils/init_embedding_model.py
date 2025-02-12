from langchain_huggingface import HuggingFaceEmbeddings
import torch

def init_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf