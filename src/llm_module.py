import os
from typing import Optional, Dict
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import openai
import cohere
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Create a .env file with your API keys:
OPENAI_API_KEY=your-key-here
COHERE_API_KEY=your-key-here
"""

@dataclass
class LLMResponse:
    """Data class to store LLM response"""
    answer: str
    model_used: str
    error: Optional[str] = None


class LLMInterface:
    """Interface for different LLM implementations"""
    
    def generate_response(
        self,
        query: str,
        context: str,
        max_tokens: int = 300
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAIInterface(LLMInterface):
    """OpenAI API implementation"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        openai.api_key = self.api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_response(
        self,
        query: str,
        context: str,
        max_tokens: int = 300
    ) -> LLMResponse:
        try:
            prompt = self._create_prompt(query, context)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return LLMResponse(
                answer=response.choices[0].message.content.strip(),
                model_used="gpt-3.5-turbo"
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return LLMResponse(
                answer="",
                model_used="gpt-3.5-turbo",
                error=str(e)
            )


class CohereInterface(LLMInterface):
    """Cohere API implementation"""
    
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        self.co = cohere.Client(self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_response(
        self,
        query: str,
        context: str,
        max_tokens: int = 300
    ) -> LLMResponse:
        try:
            prompt = self._create_prompt(query, context)
            
            response = self.co.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                model='command'
            )
            
            return LLMResponse(
                answer=response.generations[0].text.strip(),
                model_used="cohere-command"
            )
            
        except Exception as e:
            logger.error(f"Cohere API error: {str(e)}")
            return LLMResponse(
                answer="",
                model_used="cohere-command",
                error=str(e)
            )


class LocalLLMInterface(LLMInterface):
    """Local model implementation using Hugging Face transformers"""
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small but capable model
    ):
        logger.info(f"Loading local model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically choose best device (CPU/GPU)
            load_in_8bit=True   # Enable 8-bit quantization to reduce memory usage
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

    def generate_response(
        self,
        query: str,
        context: str,
        max_tokens: int = 300
    ) -> LLMResponse:
        try:
            prompt = self._create_prompt(query, context)
            
            response = self.pipeline(
                prompt,
                max_length=len(prompt) + max_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            # Extract generated text after the prompt
            generated_text = response[0]['generated_text'][len(prompt):].strip()
            
            return LLMResponse(
                answer=generated_text,
                model_used="TinyLlama-1.1B"
            )
            
        except Exception as e:
            logger.error(f"Local LLM error: {str(e)}")
            return LLMResponse(
                answer="",
                model_used="TinyLlama-1.1B",
                error=str(e)
            )


def _create_prompt(query: str, context: str) -> str:
    """Create a standardized prompt format"""
    return f"""Context: {context}

Question: {query}

Please provide a clear and concise answer based on the context above."""


def get_llm_response(
    query: str,
    context: str,
    llm_type: str = "openai",
    max_tokens: int = 300
) -> LLMResponse:
    """
    Get response from specified LLM type
    
    Args:
        query: User question
        context: Retrieved document text
        llm_type: Type of LLM to use ('openai', 'cohere', or 'local')
        max_tokens: Maximum tokens in response
        
    Returns:
        LLMResponse object containing the answer and metadata
    """
    llm_interfaces = {
        "openai": OpenAIInterface,
        "cohere": CohereInterface,
        "local": LocalLLMInterface
    }
    
    if llm_type not in llm_interfaces:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    llm = llm_interfaces[llm_type]()
    return llm.generate_response(query, context, max_tokens)


# Example usage
def main():
    # Example query and context
    query = "What are the main benefits of renewable energy?"
    context = """
    Renewable energy sources like solar and wind power offer numerous advantages.
    They produce clean energy without harmful emissions, helping to combat climate change.
    These sources are sustainable and never run out, unlike fossil fuels.
    Additionally, renewable energy projects often create local jobs and can reduce energy costs over time.
    """
    
    # Try different LLM options
    for llm_type in ["openai", "cohere", "local"]:
        try:
            print(f"\nTrying {llm_type} LLM...")
            response = get_llm_response(query, context, llm_type=llm_type)
            
            if response.error:
                print(f"Error: {response.error}")
            else:
                print(f"Model: {response.model_used}")
                print(f"Answer: {response.answer}")
                
        except Exception as e:
            print(f"Error with {llm_type}: {str(e)}")


if __name__ == "__main__":
    main()