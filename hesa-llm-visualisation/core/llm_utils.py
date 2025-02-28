from .llm.gpt_j_handler import GPTJHandler
from .llm.settings import CURRENT_CONFIG

# Initialize the GPT-J handler with current configuration
gpt_handler = GPTJHandler(config=CURRENT_CONFIG)

def generate_response(query: str) -> dict:
    """Generate response for a natural language query."""
    return gpt_handler.process_query(query)
