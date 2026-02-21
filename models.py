from pydantic import BaseModel

class TextRequest(BaseModel):
    prompt: str

class LogoRequest(BaseModel):
    brand_name: str
    style: str