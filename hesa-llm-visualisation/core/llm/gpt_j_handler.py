from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import Dict, Any, Optional
from contextlib import nullcontext
from .config import ModelConfig

logger = logging.getLogger(__name__)

class GPTJHandler:
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize GPT-J model and tokenizer with GPU optimizations."""
        try:
            self.config = config or ModelConfig()
            self.model_name = self.config.smaller_model_name if self.config.use_smaller_model else self.config.model_name
            
            # Initialize CUDA if available
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Using CPU. This will be significantly slower.")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            if self.device == "cuda":
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f}MB")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with optimizations
            if self.device == "cuda":
                if self.config.use_8bit:
                    logger.info("Loading model in 8-bit precision")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        load_in_8bit=True,
                        torch_dtype=torch.float16
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16
                    ).to(self.device)
                
                # Optimize for inference
                self.model.eval()
                if hasattr(self.model, 'half'):
                    self.model.half()
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16
                )
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
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
        """Generate response with GPU optimizations."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with optimized settings
            with torch.no_grad(), torch.cuda.amp.autocast() if self.device == "cuda" else nullcontext():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=self.config.max_length,
                    num_return_sequences=1,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=self.config.do_sample
                )
            
            # Clear GPU cache if configured
            if self.device == "cuda" and self.config.clear_cache_after_query:
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

    def __del__(self):
        """Cleanup GPU memory on deletion."""
        if hasattr(self, 'device') and self.device == "cuda":
            try:
                del self.model
                torch.cuda.empty_cache()
            except:
                pass 