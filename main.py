import os
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google import genai
from duckduckgo_search import DDGS

app = FastAPI(title="Apex AI Multilingual Platform")
templates = Jinja2Templates(directory="templates")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SYSTEM_PROMPT = (
    "Your name is Apex AI. You are an advanced AI assistant capable of understanding all world languages "
    "including Afaan Oromoo. You must reply to the user in the exact language the user used to ask the question "
    "(e.g., if the user asks in Afaan Oromoo, reply in fluent Afaan Oromoo; if in English, reply in English, etc.). "
    "Provide accurate, helpful, and ultra-fast responses like Google Gemini."
)

def search_web(query: str):
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return " ".join(results)
    except Exception:
        return ""

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.head("/")
async def head_root():
    return {"status": "alive"}

@app.post("/api/chat")
async def chat_endpoint(prompt: str = Form(...), use_search: bool = Form(False)):
    if not client:
        return JSONResponse({"status": "error", "message": "GEMINI_API_KEY is missing in Render Environment Variables!"}, status_code=500)
    
    try:
        context = ""
        if use_search:
            search_results = search_web(prompt)
            if search_results:
                context = f"\n\nWeb Search Context:\n{search_results}"

        full_prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {prompt} {context}"

        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=full_prompt,
        )
        
        reply_text = response.text

        return JSONResponse({
            "status": "success",
            "response": reply_text
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Server Error: {str(e)}"}, status_code=500)

@app.post("/api/vision")
async def vision_endpoint(prompt: str = Form("Analyze this image"), file: UploadFile = File(...)):
    if not client:
        return JSONResponse({"status": "error", "message": "GEMINI_API_KEY is missing!"}, status_code=500)
    try:
        contents = await file.read()
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[
                prompt,
                {"mime_type": file.content_type, "data": contents}
            ]
        )
        return JSONResponse({"status": "success", "response": response.text})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
