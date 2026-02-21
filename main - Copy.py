from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import TextRequest, LogoRequest
import ai_services

app = FastAPI(title="BrandCraft API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# BRAND NAME GENERATOR
# -------------------------
@app.post("/generate-brand-name")
def generate_brand_name(request: TextRequest):
    prompt = f"Generate 10 creative startup brand names for: {request.prompt}"
    result = ai_services.generate_text(prompt)
    return {"brand_names": result}


# -------------------------
# TAGLINE GENERATOR
# -------------------------
@app.post("/generate-tagline")
def generate_tagline(request: TextRequest):
    prompt = f"Create 5 catchy taglines for: {request.prompt}"
    result = ai_services.generate_text(prompt)
    return {"taglines": result}


# -------------------------
# PRODUCT DESCRIPTION
# -------------------------
@app.post("/generate-description")
def generate_description(request: TextRequest):
    prompt = f"Write a professional product description for: {request.prompt}"
    result = ai_services.generate_text(prompt)
    return {"description": result}


# -------------------------
# LOGO GENERATOR
# -------------------------
@app.post("/generate-logo")
def generate_logo(request: LogoRequest):
    prompt = f"Vector style logo for brand {request.brand_name} in {request.style} style"
    image = ai_services.generate_logo(prompt)
    return {"image_base64": image}


# -------------------------
# SENTIMENT ANALYSIS
# -------------------------
@app.post("/analyze-sentiment")
def sentiment(request: TextRequest):
    result = ai_services.analyze_sentiment(request.prompt)
    return {"sentiment": result}


# -------------------------
# SUMMARIZATION
# -------------------------
@app.post("/summarize")
def summarize(request: TextRequest):
    result = ai_services.summarize_text(request.prompt)
    return {"summary": result}