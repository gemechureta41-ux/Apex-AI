import os
import base64
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from duckduckgo_search import DDGS

app = FastAPI(title="Apex AI Full Platform")
templates = Jinja2Templates(directory="templates")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PROMPT = (
    "Maqaan kee Apex AI jedhama. Ati gargaaraa AI dandeettii ol'aanaa qabu, "
    "afaanota addunyaa fi Afaan Oromoo guutummaatti hubatu, barreeffama, sagalee, "
    "suuraa fi vidiyoo xiinxalee deebii sirrii fi qulqullina qabu kennuudha."
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
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(prompt: str = Form(...), use_search: bool = Form(False)):
    if not OPENAI_API_KEY:
        return JSONResponse({"status": "error", "message": "OPENAI_API_KEY Render irratti hin konfiguramne!"}, status_code=500)
    
    try:
        context = ""
        if use_search:
            search_results = search_web(prompt)
            if search_results:
                context = f"\n\nOdeeffannoo dabalataa Marsariitii irraa argame:\n{search_results}"

        full_prompt = prompt + context

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3
        )
        
        reply_text = response.choices[0].message.content

        # Text-to-Speech Generation
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=reply_text
        )
        audio_base64 = base64.b64encode(speech_response.content).decode('utf-8')

        return JSONResponse({
            "status": "success",
            "response": reply_text,
            "audio": audio_base64
        })
    except Exception as e:
        # Error sirriitti print gochuuf akka Internal Server Error qofti hin baane
        return JSONResponse({"status": "error", "message": f"Server Error: {str(e)}"}, status_code=500)
