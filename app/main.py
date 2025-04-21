from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid, os

from app.rag_agent import add_pdf_to_vectorstore, query_llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc chỉ cho phép frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def read_root():
    return HTMLResponse(content=open("app/frontend/index.html").read())

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    add_pdf_to_vectorstore(file_path)
    return {"message": "File uploaded and indexed!"}

@app.post("/chat/")
async def chat_with_agent(prompt: str = Form(...)):
    answer = query_llm(prompt)
    return {"answer": answer}
