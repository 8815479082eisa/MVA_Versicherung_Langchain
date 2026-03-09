import ollama

# Customize these to your Ollama server
OLLAMA_HOST = "http://141.41.32.94:41314"  # your custom IP and port

client = ollama.Client(host=OLLAMA_HOST)

model_name = "lfm2.5-thinking:1.2b"  # or whatever model you have pulled

print("Sending a simple prompt...")
response = client.generate(
    model=model_name,
    prompt="Explain in one sentence what the sun is.",
)

print("\nResponse:")
print(response["response"])
