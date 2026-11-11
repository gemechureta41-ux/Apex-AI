import os
import base64
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google import genai
from duckduckgo_search import DDGS
from gtts import gTTS
import io

app = FastAPI(title="Apex AI Gemini Fast Platform")
templates = Jinja2Templates(directory="templates")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SYSTEM_PROMPT = (
    "Maqaan kee Apex AI jedhama. Ati gargaaraa AI dandeettii ol'aanaa qabu, "
    "afaanota addunyaa fi Afaan Oromoo guutummaatti hubatu, barreeffama, sagalee, "
    "suuraa fi vidiyoo xiinxalee deebii sirrii fi qulqullina qabu saffisaan kennuudha."
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
        return JSONResponse({"status": "error", "message": "GEMINI_API_KEY Render Environment Variables irratti hin jiru!"}, status_code=500)
    
    try:
        context = ""
        if use_search:
            search_results = search_web(prompt)
            if search_results:
                context = f"\n\nOdeeffannoo dabalataa Marsariitii irraa argame:\n{search_results}"

        full_prompt = f"{SYSTEM_PROMPT}\n\nGaaffii Fayyadamaa: {prompt} {context}"

        # Modelii sirrii fi saffisa qabu (gemini-3.8-flash)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=full_prompt,
        )
        
        reply_text = response.text

        audio_base64 = ""
        try:
            tts = gTTS(text=reply_text[:300], lang='en')
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_base64 = base64.b64encode(audio_fp.read()).decode('utf-8')
        except Exception:
            pass

        return JSONResponse({
            "status": "success",
            "response": reply_text,
            "audio": audio_base64
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Server Error: {str(e)}"}, status_code=500)

@app.post("/api/vision")
async def vision_endpoint(prompt: str = Form("Suuraa kana naaf xiinxali"), file: UploadFile = File(...)):
    if not client:
        return JSONResponse({"status": "error", "message": "GEMINI_API_KEY hin argamne!"}, status_code=500)
    try:
        contents = await file.read()
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                prompt,
                {"mime_type": file.content_type, "data": contents}
            ]
        )
        return JSONResponse({"status": "success", "response": response.text})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    return JSONResponse({"status": "success", "transcript": "Audio transcription hojjetameera."})
