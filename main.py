import os
import base64
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from groq import Groq
from duckduckgo_search import DDGS
from gtts import gTTS
import io

app = FastAPI(title="Apex AI Free Platform")
templates = Jinja2Templates(directory="templates")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

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
    return templates.TemplateResponse(request, "index.html")

@app.head("/")
async def head_root():
    return {"status": "alive"}

@app.post("/api/chat")
async def chat_endpoint(prompt: str = Form(...), use_search: bool = Form(False)):
    if not client:
        return JSONResponse({"status": "error", "message": "GROQ_API_KEY Render Environment Variables irratti hin jiru!"}, status_code=500)
    
    try:
        context = ""
        if use_search:
            search_results = search_web(prompt)
            if search_results:
                context = f"\n\nOdeeffannoo dabalataa Marsariitii irraa argame:\n{search_results}"

        full_prompt = prompt + context

        # Groq Stable Free Model (llama-3.1-8b-instant)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3
        )
        
        reply_text = response.choices[0].message.content

        tts = gTTS(text=reply_text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_base64 = base64.b64encode(audio_fp.read()).decode('utf-8')

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
        return JSONResponse({"status": "error", "message": "GROQ_API_KEY hin argamne!"}, status_code=500)
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
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

@app.post("/api/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    if not client:
        return JSONResponse({"status": "error", "message": "GROQ_API_KEY hin argamne!"}, status_code=500)
    try:
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as f:
            f.write(await file.read())
            
        with open(temp_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file
            )
            
        os.remove(temp_filename)
        return JSONResponse({"status": "success", "transcript": transcript.text})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
