import os
from langchain_core.messages import HumanMessage
from main import GroqOpenAILLM

# Set the API key
os.environ["GROQ_API_KEY"] = "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"

# Test the GroqOpenAILLM class
llm = GroqOpenAILLM(api_key=os.environ["GROQ_API_KEY"])

# Test simple generation
messages = [HumanMessage(content="Explain the importance of fast language models in one sentence.")]
response = llm.invoke(messages)
print("Test Response:", response.content)

# Test _call method
prompt = "What is AI security?"
response_call = llm._call(prompt)
print("Call Response:", response_call)
