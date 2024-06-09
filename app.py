from fastapi import FastAPI,File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from data_processing import pdf_processing , vectorise , create_conversation ,chat
from fastapi.responses import JSONResponse
import os
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

#update this during deployemnt
list = ["http://localhost:5000",
        "http://localhost:8000",
        "https://localhost:5000"
       ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

conversation = None

def pipeline():
    global conversation
    
    text_chunks=pdf_processing()
    logging.info("**** PDF's Processed ****")
    vectorstore=vectorise(text_chunks)
    logging.info(f"**** Pinecone Vectors Created:{vectorstore} ****")
    conversation=create_conversation(vectorstore)
    logging.info("**** CONVERSATION MODEL CREATED ****")
    logging.info("**** CHAT STARTED KINDLY ASK QUESTIONS ****")
    

@app.get("/")
def index():
    return {"message","Server Running"}

@app.post("/upload_files")
async def start(files: List[UploadFile] = File(...)):
    UPLOAD_DIR="pdfs"
    os.makedirs(UPLOAD_DIR,exist_ok=True)
    uploaded_files=[]
    for file in files:
        file_location=os.path.join(UPLOAD_DIR,file.filename)
        with open(file_location,"wb") as f:
            f.write(await file.read())
        uploaded_files.append(file.filename)
    
    logging.info("**** FILES UPLOADED SUCCESSFULLY ****")
    pipeline()
    logging.info("**** CHAT CREATED, CLEARING PATH ****")
    
    for root,_,files in os.walk(UPLOAD_DIR):
        for name in files:
            os.remove(os.path.join(root,name))
    logging.info("**** PATH CLEARED ****")
    logging.infO("**** START CHAT ****")
    
    return JSONResponse(content={"message": "Files uploaded successfully. Chat Created", "filenames": uploaded_files})

@app.post("/chat")
def chat_start(question:str):
    global conversation
    if conversation is None:
        return JSONResponse(content={"message": "Conversation model not created yet. Upload files first."}, status_code=400)
    try:
        response=chat(conversation,question)
        logging.info(response)
        return JSONResponse(content={"response":response['answer']})
    except Exception as e:
        return JSONResponse(content={"error":"Chat Error"},status_code=400)
        
    
if __name__=="main":
    uvicorn.run(app,port=5001)
    
