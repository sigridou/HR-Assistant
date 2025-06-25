import torch
import ollama
from config import EMBED_MODEL

def generate_embeddings(content_list):
    embeddings = []
    for content in content_list:
        try:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=content)
            emb = response.get("embedding")
            if emb:
                embeddings.append(emb)
            else:
                print(f"[warn] No embedding for: {content[:50]}")
        except Exception as e:
            print(f"[error] Failed embedding: {e}")
    print(f"[info] Generated {len(embeddings)} embeddings for {len(content_list)} items.")
    return torch.tensor(embeddings) if embeddings else torch.tensor([])

def get_relevant_context(query, vault_embeddings, vault_content, top_k=10):
    try:
        if vault_embeddings.nelement() == 0:
            return []
        embedding_response = ollama.embeddings(model=EMBED_MODEL, prompt=query)
        input_embedding = embedding_response.get("embedding")
        if not input_embedding:
            print("Error: input_embedding is empty or not generated.")
            return []
        cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
        top_k = min(top_k, len(cos_scores))
        top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
        relevant_context = [vault_content[idx].strip() for idx in top_indices]
        return relevant_context
    except Exception as e:
        print(f"Error in get_relevant_context: {e}")
        return []
