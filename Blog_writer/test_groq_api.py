import openai
import os

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = input("Enter your Groq API key: ")

openai.api_key = groq_api_key
openai.api_base = "https://api.groq.com/openai/v1"

try:
    response = openai.ChatCompletion.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello, world!"}]
    )
    print("Groq API test successful! Response:")
    print(response)
except Exception as e:
    print("Groq API error:", e)
