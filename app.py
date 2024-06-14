from fastapi import FastAPI,File, UploadFile ,Request ,Form
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

origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

conversation = None

def pipeline(model):
    global conversation
    
    text_chunks=pdf_processing()
    logging.info("**** PDF's Processed ****")
    vectorstore=vectorise(text_chunks,model)
    logging.info(f"**** Pinecone Vectors Created:{vectorstore} ****")
    conversation=create_conversation(vectorstore,model)
    logging.info("**** CONVERSATION MODEL CREATED ****")
    logging.info("**** CHAT STARTED KINDLY ASK QUESTIONS ****")
    

@app.get("/")
def index():
    return {"message","Server Running"}

@app.post("/upload_files")
async def start(files:List[UploadFile]=File(...),model: str = Form(...)):
    try: 
        print("FILES:",files)
        print("MODEL:",model)
        
        if not files or not model:
            return JSONResponse(content={"error":"Kindly Upload Files and Select a model"},status_code=400)
        UPLOAD_DIR="pdfs"
        os.makedirs(UPLOAD_DIR,exist_ok=True)
        uploaded_files=[]
        for file in files:
            file_location=os.path.join(UPLOAD_DIR,file.filename)
            with open(file_location,"wb") as f:
                f.write(await file.read())
            uploaded_files.append(file.filename)
        logging.info(f"**** FILES UPLOADED {uploaded_files} ****")
        
        logging.info("**** FILES UPLOADED SUCCESSFULLY ****")
        pipeline(model)
        logging.info("**** CHAT CREATED, CLEARING PATH ****")
        
        for root,_,files in os.walk(UPLOAD_DIR):
            for name in files:
                os.remove(os.path.join(root,name))
        
        logging.info("**** PATH CLEARED ****")
        logging.info("**** START CHAT ****")
        
        # headers = {
        # "Access-Control-Allow-Origin": "*",}
        
        return JSONResponse(content={"message": "Files uploaded successfully. Chat Created", "filenames": uploaded_files},status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error":"Files not Uploaded(e)"},status_code=400)

@app.post("/chat")
async def chat_start(request: Request):
    global conversation
    data=await request.json()
    question=data.get('question')
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
    
