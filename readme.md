# Exonova

A chat bot for the Nostr network.

## Using oxChat to test the app

You must add this relay: `wss://relay.nostr.net` to 0xChat.

## Installing

(Built for python 3.12 LTS)

## (MacOS) - install python 3.12 and set up a virtual environment

    brew install python@3.12
    python3 -m venv .venv
    source .venv/bin/activate

## Vultr: Install llama cpp for cuda

    # Install wheel
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

    # Fix broken weird paths
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib/

## Install dependencies

    pip3 install -r requirements.txt

## Create a .env file with the following

    PRIVATE_KEY=<your private nsec key>
    CREATOR_PUBLIC_KEY=<your public nsec key>

## Run the app

    python3 main.py
