from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GPTJHandler:
    def __init__(self, use_8bit=True):
        """Initialize GPT-J model and tokenizer with optimizations."""
        try:
            self.model_name = "EleutherAI/gpt-j-6B"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Check CUDA availability
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load model with optimizations
            if self.device == "cuda":
                if use_8bit:
                    # Load in 8-bit precision for memory efficiency
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        load_in_8bit=True
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    self.model.to(self.device)
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                if torch.cuda.is_available():
                    logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            else:
                # CPU optimizations
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16
                )
            
        except Exception as e:
            logger.error(f"Error initializing GPT-J: {str(e)}")
            raise

    def prepare_prompt(self, query: str) -> str:
        """Prepare the prompt for GPT-J."""
        return f"""
        Convert the following natural language query about HESA data into structured parameters:
        Query: {query}
        
        Extract the following information:
        - Metrics (what to measure)
        - Time period
        - Institutions involved
        - Type of comparison
        - Required visualization
        
        Format the response as JSON.
        """

    def generate_response(self, prompt: str) -> str:
        """Generate response with optimized settings."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with optimized settings
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=100,  # Reduced from 200
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,  # Enable sampling for faster generation
                )
            
            # Clear cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured format."""
        try:
            # Extract JSON-like structure from response
            import json
            import re
            
            # Find JSON-like structure in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON structure found in response")
            
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['metrics', 'time_period', 'institutions']
            missing_fields = [field for field in required_fields if field not in parsed]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return structured parameters."""
        try:
            prompt = self.prepare_prompt(query)
            response = self.generate_response(prompt)
            parsed_response = self.parse_response(response)
            
            logger.info(f"Successfully processed query: {query}")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise 