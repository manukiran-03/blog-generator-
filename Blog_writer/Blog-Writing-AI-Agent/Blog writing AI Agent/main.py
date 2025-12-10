# Calculate read_time for blog blocks
def calculate_read_time(blocks):
    WORDS_PER_MINUTE = 200
    total_words = 0
    for block in blocks:
        btype = block.get("type")
        data = block.get("data", {})
        if btype in ["paragraph", "header"]:
            total_words += len(data.get("text", "").split())
        elif btype == "list":
            for item in data.get("items", []):
                total_words += len(item.split())
        elif btype == "quote":
            total_words += len(data.get("text", "").split())
        elif btype == "table":
            for row in data.get("content", []):
                for cell in row:
                    total_words += len(cell.split())
    minutes = max(1, (total_words + WORDS_PER_MINUTE - 1) // WORDS_PER_MINUTE)
    return f"{minutes} min read"
import re
import json

# Blog validation and formatting utility for AIPrism Blog Input Specification
def validate_and_format_blog(blog_json):
    """
    Validate and format the blog post according to AIPrism Blog Input Specification.
    Returns (valid: bool, errors: list, formatted_blog: dict)
    """
    errors = []
    # Ensure blog_json is a dict
    if not isinstance(blog_json, dict):
        errors.append("Blog output is not a valid JSON object.")
        return False, errors, blog_json
    # Required fields
    required_fields = ["title", "slug", "cover_image", "author", "tags", "excerpt", "category", "blocks"]
    for field in required_fields:
        if field not in blog_json or not blog_json[field]:
            errors.append(f"Missing required field: {field}")

    # Title
    title = blog_json.get("title", "")
    if not (10 <= len(title) <= 200):
        errors.append("Title must be between 10 and 200 characters.")

    # Slug
    slug = blog_json.get("slug", "")
    if not re.match(r"^[a-z0-9-]+$", slug):
        errors.append("Slug must contain only lowercase letters, numbers, and hyphens.")
    if not (5 <= len(slug) <= 100):
        errors.append("Slug must be between 5 and 100 characters.")

    # Cover image
    cover_image = blog_json.get("cover_image", "")
    if not (cover_image.startswith("https://") and 10 <= len(cover_image) <= 500):
        errors.append("Cover image must be a valid HTTPS URL (10-500 chars).")

    # Author
    author = blog_json.get("author", {})
    if "name" not in author or not (2 <= len(author.get("name", "")) <= 100):
        errors.append("Author name must be between 2 and 100 characters.")

    # Tags
    tags = blog_json.get("tags", [])
    if not (isinstance(tags, list) and 1 <= len(tags) <= 10):
        errors.append("Tags must be a list with 1-10 items.")
    for tag in tags:
        if not (2 <= len(tag) <= 30):
            errors.append(f"Tag '{tag}' must be between 2 and 30 characters.")

    # Excerpt
    excerpt = blog_json.get("excerpt", "")
    if not (50 <= len(excerpt) <= 500):
        errors.append("Excerpt must be between 50 and 500 characters.")

    # Category
    category = blog_json.get("category", "")
    if category not in ["Compliance", "Security", "Research", "General"]:
        errors.append("Category must be one of: Compliance, Security, Research, General.")

    # Blocks
    blocks = blog_json.get("blocks", [])
    if not (isinstance(blocks, list) and 1 <= len(blocks) <= 100):
        errors.append("Blocks must be a list with 1-100 items.")
    # Validate each block
    for i, block in enumerate(blocks):
        btype = block.get("type")
        data = block.get("data", {})
        if btype not in ["header", "paragraph", "image", "list", "table", "code", "quote"]:
            errors.append(f"Block {i+1}: Invalid type '{btype}'.")
        if btype == "header":
            if "text" not in data or not (1 <= len(data["text"]) <= 200):
                errors.append(f"Block {i+1}: Header text required, max 200 chars.")
            if "level" not in data or not (2 <= int(data["level"]) <= 6):
                errors.append(f"Block {i+1}: Header level must be 2-6.")
        elif btype == "paragraph":
            if "text" not in data or not (1 <= len(data["text"]) <= 5000):
                errors.append(f"Block {i+1}: Paragraph text required, max 5000 chars.")
        elif btype == "image":
            if "url" not in data or not data["url"].startswith("https://"):
                errors.append(f"Block {i+1}: Image URL must be valid HTTPS.")
            if "caption" in data and len(data["caption"]) > 200:
                errors.append(f"Block {i+1}: Image caption max 200 chars.")
        elif btype == "list":
            items = data.get("items", [])
            if not (isinstance(items, list) and 1 <= len(items) <= 20):
                errors.append(f"Block {i+1}: List must have 1-20 items.")
            for item in items:
                if len(item) > 500:
                    errors.append(f"Block {i+1}: List item max 500 chars.")
        elif btype == "table":
            content = data.get("content", [])
            if not (isinstance(content, list) and 2 <= len(content) <= 50):
                errors.append(f"Block {i+1}: Table must have 2-50 rows.")
            for row in content:
                if not (isinstance(row, list) and 1 <= len(row) <= 10):
                    errors.append(f"Block {i+1}: Table row must have 1-10 columns.")
                for cell in row:
                    if len(cell) > 200:
                        errors.append(f"Block {i+1}: Table cell max 200 chars.")
        elif btype == "code":
            if "code" not in data or not (1 <= len(data["code"]) <= 10000):
                errors.append(f"Block {i+1}: Code required, max 10000 chars.")
        elif btype == "quote":
            if "text" not in data or not (1 <= len(data["text"]) <= 1000):
                errors.append(f"Block {i+1}: Quote text required, max 1000 chars.")
            if "author" in data and len(data["author"]) > 100:
                errors.append(f"Block {i+1}: Quote author max 100 chars.")

    valid = len(errors) == 0
    return valid, errors, blog_json
import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from typing import List, Any
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# CrewAI removed from runtime to avoid version conflicts with langchain.

load_dotenv()

class GoogleGenerativeAILLM:
    """Custom wrapper for Google Generative AI to work with CrewAI"""
    model_name: str = "gemini-2.5-flash"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    generative_model: Any = None

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.7):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = model
        self.temperature = temperature
        self.generative_model = genai.GenerativeModel(model)

    def _call(self, prompt: str, **kwargs) -> str:
        response = self.generative_model.generate_content(prompt, generation_config={
            "temperature": self.temperature,
            "max_output_tokens": 2048,
        })
        # Robust handling for Gemini API responses
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                # finish_reason 2 means NO_CONTENT, skip it
                if getattr(candidate, "finish_reason", None) == 2:
                    continue
                if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            return part.text
        return "ERROR: No valid content generated. The model did not return any output."

    def _call(self, prompt: str, **kwargs) -> str:
        response = self.generative_model.generate_content(prompt, generation_config={
            "temperature": self.temperature,
            "max_output_tokens": 2048,
        })
        return getattr(response, "text", "")

class GroqOpenAILLM:
    """Custom wrapper for Groq using OpenAI client to work with CrewAI"""
    model_name: str = "openai/gpt-oss-20b"
    temperature: float = 0.7
    client: Any = None

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b", temperature: float = 0.7):
        super().__init__()
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model_name = model
        self.temperature = temperature

    def _call(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def _generate(self, messages: List[Any], **kwargs) -> str:
        raise NotImplementedError()

    # ...existing code...
class AnthropicLLM:
    """Custom wrapper for Anthropic Claude to work with CrewAI"""
    model_name: str = "claude-3.5-sonnet"
    temperature: float = 0.7
    client: Any = None

    def __init__(self, api_key: str, model: str = "claude-3.5-sonnet", temperature: float = 0.7):
        super().__init__()
        if Anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self.client = Anthropic(api_key=api_key)
        self.model_name = model
        self.temperature = temperature

    def _call(self, prompt: str, **kwargs) -> str:
        resp = self.client.messages.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            for block in getattr(resp, "content", []) or []:
                # Newer anthropic SDK returns blocks with .type and .text
                if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                    return block.text
            # Fallbacks
            if getattr(resp, "content", None) and hasattr(resp.content[0], "text"):
                return resp.content[0].text
        except Exception:
            pass
        return ""

    def _generate(self, messages: List[Any], **kwargs) -> str:
        raise NotImplementedError()
def generate_aiprism_blog_json(llm, topic, feedback=None):
    long_example = """{"slug":"comprehensive-ai-security-guide-2025","title":"Comprehensive AI Security Guide for 2025","author":{"id":"u001","name":"Dr. Anjan Krishnamurthy"},"cover_image":"https://images.unsplash.com/photo-1563986768609-322da13575f3?w=800","tags":["Security","Best Practices","AI Safety"],"excerpt":"A complete guide to securing AI systems in 2025, covering threat models, mitigation strategies, and compliance requirements.","category":"Security","blocks":[{"type":"header","data":{"text":"Introduction","level":2}},{"type":"paragraph","data":{"text":"In 2025, AI security has become paramount as systems grow more autonomous and integrated into critical infrastructure."}},{"type":"image","data":{"url":"https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=800","caption":"Modern AI security infrastructure"}},{"type":"header","data":{"text":"Key Security Principles","level":2}},{"type":"list","data":{"items":["Defense in Depth: Multiple layers of security controls","Zero Trust Architecture: Never trust, always verify","Continuous Monitoring: Real-time threat detection","Incident Response: Prepared recovery procedures"]}},{"type":"header","data":{"text":"Implementation Example","level":3}},{"type":"code","data":{"code":"from aiprism import SecurityScanner\n\nscanner = SecurityScanner()\nresults = scanner.scan_model('my-model.pkl')\n\nif results.has_vulnerabilities():\n    print(f'Found {results.count} issues')","language":"python"}},{"type":"header","data":{"text":"Comparison of Security Frameworks","level":2}},{"type":"table","data":{"content":[["Framework","Scope","Compliance","Adoption"],["NIST AI RMF","Comprehensive","Federal","High"],["ISO 42001","Governance","International","Growing"],["OWASP ML Top 10","Vulnerabilities","Industry","Moderate"]]}},{"type":"quote","data":{"text":"Security is not a product, but a process. In AI, this process must be continuous and adaptive.","author":"Bruce Schneier, Security Technologist"}},{"type":"header","data":{"text":"Conclusion","level":2}},{"type":"paragraph","data":{"text":"AI security in 2025 requires a holistic approach combining technical controls, governance, and continuous adaptation to emerging threats."}}]}"""
    prompt = f"""
Output only a valid JSON object for the following blog topic. Do not include any explanation or extra text.
Topic: {topic}

Example:
{long_example}
"""
    if feedback:
        prompt += f"\n\nPrevious output failed validation for these reasons: {feedback}. Please fix these issues and try again."
    return llm._call(prompt)

# Initialize LLM based on provider
def initialize_llm(provider):
    if provider == "Groq":
        groq_api_key = (os.getenv("GROQ_API_KEY", "").strip() or st.secrets.get("GROQ_API_KEY", "").strip())
        if not groq_api_key:
            st.error("Please set your GROQ_API_KEY environment variable")
            return None

        return GroqOpenAILLM(
            api_key=groq_api_key,
            model="openai/gpt-oss-20b",
            temperature=0.7
        )
    elif provider == "Google AI Studio":
        google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("Please set your GOOGLE_API_KEY environment variable")
            return None

        return GoogleGenerativeAILLM(
            api_key=google_api_key,
            model="gemini-2.5-flash",
            temperature=0.7
        )
    elif provider == "Anthropic":
        # Interpret requested "Claude Sonnet 4.5" as latest Sonnet; default to Claude 3.5 Sonnet
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            st.error("Please set your ANTHROPIC_API_KEY environment variable")
            return None
        # If user provided a specific override via env, use it; else default
        model = os.getenv("ANTHROPIC_MODEL", "claude-3.5-sonnet")
        return AnthropicLLM(
            api_key=anthropic_api_key,
            model=model,
            temperature=0.7,
        )
    else:
        st.error("Invalid LLM provider selected")
        return None

# Initialize tools
def initialize_tools():
    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()
    return [search_tool, scrape_tool]

class AISecurityBlogCrew:
    def __init__(self, provider):
        self.llm = initialize_llm(provider)
        self.tools = initialize_tools()
        
    def create_research_agent(self):
        return Agent(
            role='AI Security Research Specialist',
            goal='Conduct comprehensive research on AI security topics',
            backstory="""Expert in ML vulnerabilities, adversarial attacks, and LLM security.
            Specializes in finding credible sources and analyzing technical papers.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm,
            max_iter=100
        )
    
    def create_writer_agent(self):
        return Agent(
            role='AI Security Content Writer',
            goal='Transform research into engaging blog posts',
            backstory="""Technical writer specializing in making complex security concepts
            accessible and engaging for technical audiences.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_iter=100
        )
    
    def create_editor_agent(self):
        return Agent(
            role='Technical Blog Editor',
            goal='Polish content for clarity, accuracy and SEO',
            backstory="""Meticulous editor with expertise in technical content and
            digital publishing. Ensures technical accuracy and readability.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_iter=100,
            max_execution_time=300
        )
    
    # ...existing code...

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Security Blog Generator",
        page_icon="🔒",
        layout="wide"
    )
    
    st.title("🔒 AI Security Blog Generator")
    
    # Configuration
    with st.sidebar:
        st.header("Configuration")

        # LLM Provider Selection
        llm_provider = st.selectbox(
            "Select LLM Provider",
            ["Groq", "Anthropic", "Google AI Studio"],
            help="Choose the AI model provider for blog generation"
        )

        if llm_provider == "Groq":
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=os.getenv("GROQ_API_KEY", ""),
                help="Get from https://console.groq.com/"
            )
            if groq_api_key:
                groq_api_key = groq_api_key.strip()
                os.environ["GROQ_API_KEY"] = groq_api_key
            st.info("Using Groq model: openai/gpt-oss-20b")
        elif llm_provider == "Anthropic":
            anthropic_api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                help="Get from https://console.anthropic.com/"
            )
            if anthropic_api_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key.strip()
            st.info("Using Anthropic Claude Sonnet (default: claude-3.5-sonnet)")
        elif llm_provider == "Google AI Studio":
            google_api_key = st.text_input(
                "Google AI Studio API Key",
                type="password",
                value=os.getenv("GOOGLE_API_KEY", ""),
                help="Get from https://aistudio.google.com/app/apikey"
            )
            if google_api_key:
                os.environ["GOOGLE_API_KEY"] = google_api_key
            st.info("Using Google Gemini 1.5 Flash model")

        # Serper API Key
        serper_api_key = st.text_input(
            "Serper API Key",
            type="password",
            value=os.getenv("SERPER_API_KEY", ""),
            help="Enter your Serper API key for web search. Get one from https://serper.dev/"
        )
        if serper_api_key:
            os.environ["SERPER_API_KEY"] = serper_api_key

    # Main content
    topic = st.text_area(
        "Enter AI security topic:",
        placeholder="e.g., 'Adversarial attacks on LLMs', 'AI model poisoning techniques'",
        height=100
    )
    
    sample_topics = [
        "Prompt injection attacks",
        "AI model stealing techniques",
        "Privacy risks in federated learning",
        "Defending against adversarial examples"
    ]
    
    if st.button("Use Sample Topic"):
        topic = sample_topics[0]
    
    if st.button("Generate Blog Post", disabled=not topic):
        if llm_provider == "Groq" and not os.getenv("GROQ_API_KEY"):
            st.error("Missing Groq API Key")
            return
        elif llm_provider == "Google AI Studio" and not os.getenv("GOOGLE_API_KEY"):
            st.error("Missing Google AI Studio API Key")
            return
        elif llm_provider == "Anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            st.error("Missing Anthropic API Key")
            return

        with st.spinner("Creating your AI security blog post..."):
            try:
                llm = initialize_llm(llm_provider)
                result = generate_aiprism_blog_json(llm, topic)

                # Robust JSON extraction from LLM output
                def extract_json(text):
                    import re, json
                    # Try to extract the largest JSON object from the output
                    matches = re.findall(r'({[\s\S]*})', text)
                    for m in matches[::-1]:  # Try last match first
                        try:
                            return json.loads(m)
                        except Exception:
                            continue
                    # Try to extract a JSON array if present
                    matches = re.findall(r'(\[[\s\S]*\])', text)
                    for m in matches[::-1]:
                        try:
                            return json.loads(m)
                        except Exception:
                            continue
                    # Fallback: try to parse the whole text
                    try:
                        return json.loads(text)
                    except Exception:
                        return None

                blog_json = extract_json(result) if isinstance(result, str) else result
                if not blog_json:
                    st.error("Blog output is not valid JSON. Please check your LLM output format.")
                    st.info("Raw LLM output:")
                    st.code(result)
                    return

                max_retries = 2
                retries = 0
                while True:
                    valid, errors, formatted_blog = validate_and_format_blog(blog_json)
                    if valid:
                        break
                    if retries >= max_retries:
                        st.error("Blog validation failed after multiple attempts. Please fix the following errors:")
                        for err in errors:
                            st.write(f"- {err}")
                        st.info("Raw LLM output:")
                        st.code(result)
                        return
                    # Retry with feedback
                    feedback = "; ".join(errors)
                    result = generate_aiprism_blog_json(llm, topic, feedback=feedback)
                    blog_json = extract_json(result) if isinstance(result, str) else result
                    retries += 1

                # Auto-calculate and insert read_time
                formatted_blog["read_time"] = calculate_read_time(formatted_blog.get("blocks", []))

                st.success("Blog Generated Successfully!")
                st.subheader("Final Blog Post (JSON)")
                st.json(formatted_blog)
                blog_title = formatted_blog.get("title", "ai_security_blog")
                safe_title = "_".join(blog_title.lower().split())
                st.download_button(
                    label="Download Blog JSON",
                    data=json.dumps(formatted_blog, indent=2),
                    file_name=f"{safe_title}.json"
                )

            except Exception as e:
                import traceback
                st.error(f"Error: {str(e)}")
                st.info("Traceback:")
                st.code(traceback.format_exc())
                st.info("Ensure you have compatible CrewAI versions: pip install crewai==0.28.8 crewai-tools==0.1.6")

if __name__ == "__main__":
    main()