import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "phi3",
        "prompt": "Suggest healthy lunch ideas",
        "stream": False
    }
)

print(response.json()["response"])