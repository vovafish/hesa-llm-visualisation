from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model optimization."""
    # Model selection
    model_name: str = "EleutherAI/gpt-j-6B"
    use_smaller_model: bool = False  # Set to True to use smaller model
    smaller_model_name: str = "EleutherAI/gpt-neo-125M"  # 125M parameters instead of 6B
    
    # Memory optimizations
    use_8bit: bool = True  # 8-bit quantization
    use_4bit: bool = False  # 4-bit quantization (even more memory efficient)
    low_cpu_mem_usage: bool = True
    
    # Generation settings
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Cache settings
    clear_cache_after_query: bool = True
    
    @classmethod
    def get_development_config(cls):
        """Get configuration optimized for development/testing."""
        return cls(
            use_smaller_model=True,
            use_8bit=True,
            max_length=50,
            temperature=0.9
        )
    
    @classmethod
    def get_production_config(cls):
        """Get configuration optimized for production."""
        return cls(
            use_smaller_model=False,
            use_8bit=True,
            max_length=100,
            temperature=0.7
        ) 