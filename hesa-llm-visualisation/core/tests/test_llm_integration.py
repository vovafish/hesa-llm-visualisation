from django.test import TestCase
from core.llm_utils import generate_response
from core.utils.query_processor import parse_llm_response

class LLMIntegrationTests(TestCase):
    def test_response_generation(self):
        query = "Show me student numbers after 2020"
        response = generate_response(query)
        self.assertIsNotNone(response)
