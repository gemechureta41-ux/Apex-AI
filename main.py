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
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "Maqaan kee Apex AI jedhama. Ati gargaaraa AI dandeettii ol'aanaa qabu, "
    "afaanota addunyaa fi Afaan Oromoo guutummaatti hubatu, barreeffama, sagalee, "
    "suuraa fi vidiyoo xiinxalee deebii sirrii fi qulqullina qabu kennuudha. "
    "Yeroo odeeffannoo yeroo ammaa barbaachisu, ragaalee dhihaatan fayyadamuun deebisi."
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

# 1. Chat with Web Search & Text-to-Speech Generation
@app.post("/api/chat")
async def chat_endpoint(prompt: str = Form(...), use_search: bool = Form(False)):
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

        # Text-to-Speech (TTS): Deebii barreeffamaa gara Sagaleetti jijjiiruu
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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# 2. Vision (Image Processing) API
@app.post("/api/vision")
async def vision_endpoint(prompt: str = Form("Suuraa kana naaf xiinxali"), file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        )
        return JSONResponse({"status": "success", "response": response.choices[0].message.content})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# 3. Audio Transcribe (Speech-to-Text) API
@app.post("/api/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    try:
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as f:
            f.write(await file.read())
            
        with open(temp_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            
        os.remove(temp_filename)
        return JSONResponse({"status": "success", "transcript": transcript.text})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
