import os
from main import AISecurityBlogCrew

# Set the API key
os.environ["GROQ_API_KEY"] = "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"

# Test blog generation
print("Testing blog generation with Groq integration...")
blog_crew = AISecurityBlogCrew("Groq")
result = blog_crew.run_blog_creation("Prompt injection attacks")

print("Blog generation completed!")
print("Result length:", len(result))
print("First 500 characters:")
print(result[:500])
print("...")
print("Last 500 characters:")
print(result[-500:])
