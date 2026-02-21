import os
from groq import Groq
from dotenv import load_dotenv
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import base64

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------
# TEXT GENERATION (Groq LLaMA)
# -------------------------
def generate_text(prompt: str):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content


# -------------------------
# SENTIMENT ANALYSIS
# -------------------------
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text: str):
    result = sentiment_pipeline(text)
    return result


# -------------------------
# SUMMARIZATION
# -------------------------
summarizer = pipeline("summarization")

def summarize_text(text: str):
    result = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return result[0]["summary_text"]


# -------------------------
# LOGO GENERATION (SDXL)
# -------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_logo(prompt: str):
    image = pipe(prompt).images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str