"""
Baseline prompt strategy. The autoresearch driver agent will edit this file
to discover better strategies. The function signature must remain stable.
"""
from google import genai
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)

def image_to_prompt(image: Image.Image) -> str:
    """Given a reference image, return a prompt for Nano Banana 2."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    response = _client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=[
            {"role": "user", "parts": [
                {"text": "Describe this image so it can be regenerated."},
                {"inline_data": {"mime_type": "image/png", "data": buf.getvalue()}},
            ]},
        ],
    )
    return response.text.strip()
