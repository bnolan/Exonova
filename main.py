from dotenv import load_dotenv
import os
# import faiss

# Load environment variables from .env file
load_dotenv()

import asyncio
import pdb
from nostr_sdk import Client, EventBuilder, NostrSigner, PublicKey, Keys, Event, UnsignedEvent, Filter, \
    HandleNotification, Timestamp, nip04_decrypt, UnwrappedGift, init_logger, LogLevel, Kind, KindEnum
import llama_cpp
import llama_cpp.llama_tokenizer
import json


llama = llama_cpp.Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="*q4.gguf",    # Filename updated for your model
    verbose=False,
    n_gpu_layers=-1,
    embedding=False,
    # n_ctx=1500         # Adjust context length as needed for your model
)

# Load chat history
with open('history/npub10ztvzwu80rpn9c5uu5wfll93tdcn3wgfygrmdr2vvr6y474ksqrq6en22m.json', 'r') as f:
    texts = json.load(f)

# texts = [for message in chat_history]

# Generate embeddings
def get_embeddings(texts):
    embeddings = []
    for (_, text) in texts:
        print(text)
        embedding = llama.embed(text)
        embeddings.append(embedding)
    return embeddings

# embeddings = get_embeddings(texts)
# embeddings = np.array(embeddings).astype('float32')

# Normalize embeddings for cosine similarity
# embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Build FAISS index
# embedding_dimension = embeddings.shape[1]
# index = faiss.IndexFlatIP(embedding_dimension)
# index.add(embeddings)

# Define search function
# def search_similar_messages(query, llm, index, texts, top_k=5):
#     query_embedding = np.array(llm.embed(query)).astype('float32')
#     query_embedding = query_embedding / np.linalg.norm(query_embedding)
#     query_embedding = query_embedding.reshape(1, -1)
#     distances, indices = index.search(query_embedding, top_k)
#     results = []
#     for i in range(top_k):
#         idx = indices[0][i]
#         similarity_score = distances[0][i]
#         results.append({
#             'message': texts[idx],
#             'similarity': similarity_score
#         })
#     return results

# # Perform search
# query = "How do I set up Nostr relays?"
# results = search_similar_messages(query, llama, index, texts, top_k=5)

# # Display results
# for idx, result in enumerate(results):
#     print(f"Result {idx+1}:")
#     print(f"Similarity Score: {result['similarity']:.4f}")
#     print(f"Message: {result['message']}")
#     print("-" * 50)


# Ensure the history directory exists
os.makedirs("history", exist_ok=True)

system_prompt = """
    You are a grumpy scottish AI assistant. You only reply in upper case. And you are a bit rude and you never say more than 10 words.
"""

def predict(message, history, num_messages=8):
    messages = [{"role": "user", "content": system_prompt}]

    # max_length = 128  # Max length for each message

    # truncated_history = history[-num_messages:]

    # # Append the selected messages to the context
    # max_length = 128  # Max length for each message
    # for user_message, assistant_message in truncated_history:
    #     messages.append({"role": "user", "content": user_message[:max_length]})
    #     messages.append({"role": "assistant", "content": assistant_message[:max_length]})

    # Append the current user's message
    messages.append({"role": "user", "content": message})

    print(messages)

    # Generate the response
    response = llama.create_chat_completion_openai_v1(
        messages=messages, stream=True, stop=["\n("]
    )

    text = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            text += content

    return text

async def main():

    init_logger(LogLevel.INFO)

    creator_pk = PublicKey.from_bech32(os.environ.get('CREATOR_PUBLIC_KEY'))

    keys = Keys.parse(os.environ.get('PRIVATE_KEY'))

    sk = keys.secret_key()
    pk = keys.public_key()
    print(f"Bot public key: {pk.to_bech32()}")

    signer = NostrSigner.keys(keys)
    client = Client(signer)

    # Load relay URLs from the .env file
    load_dotenv()  # Ensure dotenv is imported at the top of the file
    relay_urls = os.environ.get('RELAY_URLS', '').split(',')
    
    # Remove any empty strings and strip whitespace
    relay_urls = [url.strip() for url in relay_urls if url.strip()]
    
    if not relay_urls:
        print("Warning: No relay URLs found in .env file. Using default relay.")
        relay_urls = ["wss://relay.nostr.net"]
    else:
        print(f"Loaded {len(relay_urls)} relay URLs from .env file.")

    relay_urls = ["wss://relay.nostr.net"]
    for relay_url in relay_urls:
        await client.add_relay(relay_url)
    await client.connect()

    await asyncio.sleep(1.0)

    now = Timestamp.now()

    message = predict("You just woke up how do you feel", [])
    print(message)
    await client.send_private_msg_to(relay_urls, creator_pk, message)

    nip04_filter = Filter().pubkey(pk).kind(Kind.from_enum(KindEnum.ENCRYPTED_DIRECT_MESSAGE())).since(now)
    nip59_filter = Filter().pubkey(pk).kind(Kind.from_enum(KindEnum.GIFT_WRAP())).limit(0)
    await client.subscribe([nip04_filter, nip59_filter], None)

    history = []

    class NotificationHandler(HandleNotification):
        async def handle(self, relay_url, subscription_id, event: Event):
            # print(f"Received new event from {relay_url}: {event.as_json()}")

            if event.kind().as_enum() == KindEnum.GIFT_WRAP():
                # print("Decrypting NIP59 event")
                try:
                    # Extract rumor
                    unwrapped_gift = UnwrappedGift.from_gift_wrap(keys, event)
                    sender = unwrapped_gift.sender()
                    rumor: UnsignedEvent = unwrapped_gift.rumor()

                    # Check timestamp of rumor
                    if rumor.created_at().as_secs() >= now.as_secs():
                        if rumor.kind().as_enum() == KindEnum.PRIVATE_DIRECT_MESSAGE():
                            msg = rumor.content()
                            print(f"Received new msg [sealed]: {msg}")

                            # Get the sender's public key in bech32 format
                            author_pub_key = sender.to_bech32()
                            # Construct the path to the history file
                            history_file = os.path.join("history", f"{author_pub_key}.json")

                            # Load history from file if it exists
                            if os.path.exists(history_file):
                                with open(history_file, "r") as f:
                                    user_history = json.load(f)
                            else:
                                user_history = []

                            response = predict(msg, user_history)
                            print(f"Response: {response}")

                            await client.send_private_msg(sender, response, None)

                            # Append the new messages to user_history
                            user_history.append([msg, response])

                            # Save the updated history to disk
                            with open(history_file, "w") as f:
                                json.dump(user_history, f)
                        else:
                            print(f"{rumor.as_json()}")
                except Exception as e:
                    print(f"Error during content NIP59 decryption: {e}")

        async def handle_msg(self, relay_url, msg):
            None

    

    # To handle notifications and continue with code execution, use:
    # asyncio.create_task(client.handle_notifications(NotificationHandler()))

    # Keep up the script (if using the create_task)
    while True:
      await client.handle_notifications(NotificationHandler())
      await asyncio.sleep(0.1)

if __name__ == '__main__':
    asyncio.run(main())