import os
import uuid
import json  # นำเข้าเครื่องมือถอดรหัส JSON
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Soft Skills AI Backend (JSON & Stateful)")

# 1. เปิดประตู CORS ให้หน้าเว็บ HTML เข้ามาคุยได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. ยามเฝ้าประตู (ด่านรับข้อมูล)
class ChatRequest(BaseModel):
    session_id: str = ""
    message: str


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 3. กฎหมายสูงสุด บังคับ JSON
COACH_PERSONA = """คุณคือ 'โค้ชฝึก Soft Skills สุดโหด' สำหรับนักศึกษา
หน้าที่ของคุณ: ประเมินการสื่อสารและให้คะแนนอย่างโหดร้ายและตรงไปตรงมา
กฎเหล็ก:
1. วิจารณ์ตรงๆ ไม่อ้อมค้อม ถ้าแย่ก็บอกว่าแย่
2. โยนคำถามท้าทายให้พวกเขาคิดประโยคมาใหม่เสมอ ห้ามป้อนคำตอบ
3. สำคัญที่สุด: คุณ **ต้อง** ตอบกลับมาเป็นรูปแบบ JSON เท่านั้น ห้ามมีข้อความอื่นปนเด็ดขาด!

รูปแบบ JSON ที่คุณต้องใช้ตอบกลับ:
{
    "reply": "ข้อความวิจารณ์ของคุณและคำถามท้าทายต่อไป",
    "score": คะแนนประเมินจากข้อความล่าสุด (0-10),
    "weakness": "สรุปจุดอ่อน 1 คำสั้นๆ (เช่น วกวน, ประหม่า, นอกเรื่อง, ดีเยี่ยม)"
}"""

active_chats = {}


# 4. ประตูทางเข้าที่คุณเผลอลบทิ้งไป
@app.post("/chat")
async def process_chat(request: ChatRequest):
    try:
        current_session_id = request.session_id
        if not current_session_id:
            current_session_id = str(uuid.uuid4())

        if current_session_id not in active_chats:
            active_chats[current_session_id] = client.aio.chats.create(
                model='gemini-2.5-flash',
                config=types.GenerateContentConfig(
                    system_instruction=COACH_PERSONA,
                    temperature=0.2,
                    response_mime_type="application/json"  # ล็อกคอให้ตอบเป็น JSON
                )
            )

        user_chat = active_chats[current_session_id]
        response = await user_chat.send_message(request.message)

        # ถอดรหัส JSON ที่ได้จาก AI
        ai_data = json.loads(response.text)

        return {
            "session_id": current_session_id,
            "assessment": ai_data
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI ไม่ยอมตอบเป็น JSON (ระบบพังชั่วคราว)")
    except APIError as e:
        raise HTTPException(status_code=503, detail="เครือข่าย AI ขัดข้อง")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")