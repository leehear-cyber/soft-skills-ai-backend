from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("ไม่พบ GEMINI_API_KEY กรุณาตั้งค่า Environment Variable")

genai.configure(api_key=API_KEY)

# บังคับให้ AI คายผลลัพธ์เป็น JSON
model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})


# เพิ่มตัวแปร scenario เข้ามาในระบบรับข้อมูล
class ChatRequest(BaseModel):
    session_id: str
    message: str
    scenario: str


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    # Prompt แบบ Professional ที่คุณต้องการ
    system_instruction = f"""
    คุณคือ "ที่ปรึกษาด้านการสื่อสารและ Soft Skills มืออาชีพ" 
    ผู้ใช้งานจะส่งข้อความหรือ SOP มาให้คุณประเมินในสถานการณ์: [{req.scenario}]

    กฎการวิเคราะห์แบบเข้มงวด:
    1. รูปแบบ: Professional กึ่งทางการ เป็นมิตร ให้คำปรึกษาเพื่อพัฒนา ห้ามตอกย้ำจุดผิดแบบรุนแรงเหมือนด่า
    2. การให้คะแนน: ชี้จุดเด่นและจุดด้อยออกมาเป็นข้อๆ อย่างชัดเจน
    3. จุดเด่น: ต้องระบุด้วยว่าดีมากน้อยแค่ไหน และให้คะแนนย่อยในจุดเด่นนั้น
    4. จุดด้อยและวิธีแก้: เสนอวิธีการปรับปรุงโดยใช้ศัพท์กึ่งทางการ เข้าใจง่าย นำไปใช้ได้จริง
    5. โฟกัสหลัก: เน้นการวิเคราะห์ที่ "วิธีการพูด/การพิมพ์" หรือ "โครงสร้างการนำเสนอ" (การสะกดคำผิดให้แจ้งได้ แต่ไม่นำมาหักคะแนนหลัก)

    โครงสร้างการตอบกลับ (ต้องเป็น JSON เท่านั้น ห้ามตอบอย่างอื่น):
    {{
        "score": "คะแนนรวม/10",
        "weakness": "สรุปจุดด้อยหลักสั้นๆ 1-2 คำ",
        "reply": "เขียนคำแนะนำฉบับเต็ม แบ่งเป็น:\n1. ภาพรวม\n2. 🟢 จุดเด่น (พร้อมบอกระดับ/คะแนนย่อย)\n3. 🔴 จุดที่ควรพัฒนา\n4. 💡 คำแนะนำเพื่อปรับปรุง"
    }}

    ข้อความของผู้ใช้งาน: {req.message}
    """

    try:
        response = model.generate_content(system_instruction)
        result = json.loads(response.text)
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="AI ประมวลผลผิดพลาด")