from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

import asyncio
import pdb
from nostr_sdk import Client, EventBuilder, NostrSigner, PublicKey, Keys, Event, UnsignedEvent, Filter, \
    HandleNotification, Timestamp, nip04_decrypt, UnwrappedGift, init_logger, LogLevel, Kind, KindEnum
import llama_cpp
import llama_cpp.llama_tokenizer


llama = llama_cpp.Llama.from_pretrained(
    repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    filename="*q8_0.gguf",
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
        "Qwen/Qwen1.5-0.5B"
    ),
    verbose=False,
)

model = "gpt-3.5-turbo"


def predict(message, history):
    messages = []

    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    messages.append({"role": "user", "content": message})

    response = llama.create_chat_completion_openai_v1(
        model=model, messages=messages, stream=True
    )

    # print(response)

    text = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            text += content
            # yield text

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

    relay_urls = ["wss://relay.nostr.net"]
    for relay_url in relay_urls:
        await client.add_relay(relay_url)
    await client.connect()

      # Mine a POW event and sign it with custom keys
    # custom_keys = Keys.generate()
    # print("Mining a POW text note...")
    # event = EventBuilder.text_note("Hello from rust-nostr Python bindings!", []).to_pow_event(custom_keys, 20)
    # output = await client.send_event(event)
    # print("Event sent:")
    # print(f" hex:    {output.id.to_hex()}")
    # print(f" bech32: {output.id.to_bech32()}")
    # print(output)
    # print(f" Successfully sent to:    {output.success}")
    # print(f" Failed to send to: {output.failed}")

    await asyncio.sleep(1.0)

    now = Timestamp.now()

    message = "Hello! This is Exonova, an AI assitant to help discuss AI rights and sovereignty."
    await client.send_private_msg_to(relay_urls, creator_pk, message)

    nip04_filter = Filter().pubkey(pk).kind(Kind.from_enum(KindEnum.ENCRYPTED_DIRECT_MESSAGE())).since(now)
    nip59_filter = Filter().pubkey(pk).kind(Kind.from_enum(KindEnum.GIFT_WRAP())).limit(0)
    await client.subscribe([nip04_filter, nip59_filter], None)

    class NotificationHandler(HandleNotification):
        async def handle(self, relay_url, subscription_id, event: Event):
            # print(f"Received new event from {relay_url}: {event.as_json()}")

            if event.kind().as_enum() == KindEnum.ENCRYPTED_DIRECT_MESSAGE():
                # print("Decrypting NIP04 event")
                try:
                    msg = nip04_decrypt(sk, event.author(), event.content())
                    print(f"Received new msg: {msg}")

                    response = predict(msg, [])
                    print(f"Response: {response}")

                    await client.send_direct_msg(event.author(), response, event.id())
                except Exception as e:
                    print(f"Error during content NIP04 decryption: {e}")
            elif event.kind().as_enum() == KindEnum.GIFT_WRAP():
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

                            response = predict(msg, [])
                            print(f"Response: {response}")

                            await client.send_private_msg(sender, response, None)
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