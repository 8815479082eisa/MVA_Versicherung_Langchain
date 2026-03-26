import ollama

# Customize these to your Ollama server
OLLAMA_HOST = "http://141.41.32.94:41314"  # your custom IP and port

client = ollama.Client(host=OLLAMA_HOST)

model_name = "rnj-1:8b"  # keep smoke tests fast while still exercising a general-purpose model

print("Sending a simple prompt...")
response = client.generate(
    model=model_name,
    prompt="Explain in one sentence what the sun is.",
)

print("\nResponse:")
print(response["response"])
