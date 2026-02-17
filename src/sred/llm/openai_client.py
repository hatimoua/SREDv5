import base64
from openai import OpenAI
from sred.config import settings
from sred.logging import logger

# Initialize client lazily or globally? Globally is fine.
# Note: settings.OPENAI_API_KEY is SecretStr, so we need .get_secret_value() if passed explicitly, 
# but OpenAI client picks up env var OPENAI_API_KEY automatically if set. 
# However, we allow setting via .env file loaded by pydantic-settings.
api_key = settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None
client = OpenAI(api_key=api_key)

def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_vision_completion(image_path: str, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    Call OpenAI Vision model with an image.
    Returns the content string.
    """
    base64_image = encode_image(image_path)
    
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL_VISION,
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=4096, # Reasonable limit for page extraction
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI Vision API call failed: {e}")
        raise

def get_chat_completion(prompt: str, system_prompt: str = "You are a helpful assistant.", json_mode: bool = False) -> str:
    """
    Call OpenAI Chat model.
    """
    try:
        kwargs = {
            "model": settings.OPENAI_MODEL_STRUCTURED if json_mode else settings.OPENAI_MODEL_AGENT,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI Chat API call failed: {e}")
        raise
