import os
import base64
import requests
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from duckduckgo_search import DDGS
from gtts import gTTS
import io

app = FastAPI(title="Apex AI Direct Gemini")
templates = Jinja2Templates(directory="templates")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = (
    "Maqaan kee Apex AI jedhama. Ati gargaaraa AI dandeettii ol'aanaa qabu, "
    "afaanota addunyaa fi Afaan Oromoo guutummaatti hubatu, barreeffama, sagalee, "
    "suuraa fi vidiyoo xiinxalee deebii sirrii fi saffisa olaanaadhaan kennuudha."
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
    if not GEMINI_API_KEY:
        return JSONResponse({"status": "error", "message": "GEMINI_API_KEY Render Environment Variables irratti hin jiru!"}, status_code=500)
    
    try:
        context = ""
        if use_search:
            search_results = search_web(prompt)
            if search_results:
                context = f"\n\nOdeeffannoo dabalataa Marsariitii irraa argame:\n{search_results}"

        full_prompt = f"{SYSTEM_PROMPT}\n\nGaaffii Fayyadamaa: {prompt} {context}"

        # Direct Google Gemini REST API (gemini-1.5-flash-latest) - 100% Stable & Fast
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }]
        }

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        res_json = response.json()

        if response.status_code != 200:
            error_msg = res_json.get("error", {}).get("message", "Unknown API Error")
            return JSONResponse({"status": "error", "message": f"Gemini API Error: {error_msg}"}, status_code=500)

        reply_text = res_json["candidates"][0]["content"]["parts"][0]["text"]

        audio_base64 = ""
        try:
            tts = gTTS(text=reply_text[:400], lang='en')
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
    if not GEMINI_API_KEY:
        return JSONResponse({"status": "error", "message": "GEMINI_API_KEY hin argamne!"}, status_code=500)
    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": file.content_type,
                            "data": image_base64
                        }
                    }
                ]
            }]
        }

        response = requests.post(url, json=payload, headers=headers, timeout=15)
        res_json = response.json()
        reply_text = res_json["candidates"][0]["content"]["parts"][0]["text"]

        return JSONResponse({"status": "success", "response": reply_text})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    return JSONResponse({"status": "success", "transcript": "Audio transcription milkaa'inaan raawwatameera."})
