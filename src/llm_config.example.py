# LLM Configuration Example
# Copy this file to llm_config.py and customize for your environment
# llm_config.py is gitignored for local customization

# =============================================================================
# Option 1: UF Hypergator API (default for most users)
# =============================================================================
LLM_BASE_URL = "https://api.ai.it.ufl.edu/v1"
LLM_MODEL = "gpt-oss-120b"

# =============================================================================
# Option 2: Local vLLM server (for users with local GPU access)
# =============================================================================
# LLM_BASE_URL = "http://localhost:8000/v1"
# LLM_MODEL = "/blue/ytang1/models/gpt-oss-120b"

# =============================================================================
# Option 3: OpenAI API (requires OPENAI_API_KEY in .env)
# =============================================================================
# LLM_BASE_URL = None  # None = use OpenAI default
# LLM_MODEL = "gpt-4o-2024-11-20"
