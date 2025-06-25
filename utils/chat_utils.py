import ollama
from utils import embedding_utils

def ollama_chat(user_input, system_message, vault_embeddings, vault_content, model, conversation_history):
    try:
        conversation_history.append({"role": "user", "content": user_input})

        relevant_context = embedding_utils.get_relevant_context(user_input, vault_embeddings, vault_content)
        if relevant_context:
            context_str = "\n".join(relevant_context)
        else:
            print("[info] No relevant context found — query handled without vault context.")
            context_str = ""

        user_input_with_context = user_input
        if context_str:
            user_input_with_context += "\n\nRelevant Context:\n" + context_str

        conversation_history[-1]["content"] = user_input_with_context

        messages = [{"role": "system", "content": system_message}] + conversation_history

        response = ollama.chat(
            model=model,
            messages=messages,
            options={"num_predict": 2000}
        )

        conversation_history.append({"role": "assistant", "content": response['message']['content']})
        return response['message']['content']

    except Exception as e:
        print(f"Error: {e}")
        return "Error processing your request."
