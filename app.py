from fastapi import FastAPI,File, UploadFile ,Request ,Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from data_processing import pdf_processing , vectorise , create_conversation ,chat,del_vectors
from fastapi.responses import JSONResponse
import os
from typing import List
import logging
logging.basicConfig(level=logging.INFO)
import uuid
import shutil
from supabase import create_client,Client
from dotenv import load_dotenv
import json
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
load_dotenv(override=True)

supabase: Client = create_client(os.getenv("SUPABASE_URL"),os.getenv("SUPABASE_KEY"))

app = FastAPI()


origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

sessions={}

def pipeline(model,session_id):
    global conversation
    text_chunks=pdf_processing()
    logging.info("**** PDF's Processed ****")
    vectorstore=vectorise(text_chunks,model,session_id)
    logging.info(f"**** Pinecone Vectors Created:{vectorstore} ****")
    conversation=create_conversation(vectorstore,model)
    logging.info("**** CONVERSATION MODEL CREATED ****")
    logging.info("**** CHAT STARTED KINDLY ASK QUESTIONS ****")
    
    return conversation
    


@app.get("/create_session")
async def create_session():
    try:
        session_id=str(uuid.uuid4())
        sessions[session_id]={"model":None,"conversation":None}
        current_time = datetime.utcnow().isoformat()
        supabase.table('sessions').insert({"session_id":session_id,"logged_in": current_time}).execute()
        
        
        print(f"**** SESSION {session_id} CREATED ****")
        logging.info(f"**** SESSION CREATED ****")
        return JSONResponse(content={"message": "Session Created", "session_id": session_id},status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error":"Error Creating Session"},status_code=400)    

@app.post("/delete_session")
async def delete_session(request: Request):
    data=await request.json()
    session_id=data.get('session_id')
    session=supabase.table('sessions').select('*').eq('session_id',session_id).execute().data
    if not session:
        print(f"**** SESSION {session_id} NOT FOUND ****")
        logging.info(f"**** SESSION NOT FOUND ****")
        return JSONResponse(content={"error":"Session not found"},status_code=404)
    else:
        model=session[0]['model']
        
        if(model):
            del_vectors(model,session_id)
        #supabase.table('sessions').delete().eq('session_id', session_id).execute()
        if(session_id in sessions):
            current_time = datetime.utcnow().isoformat()
            supabase.table('sessions').update({"logged_out": current_time}).eq('session_id', session_id).execute()
            del sessions[session_id]
            print(f"**** SESSION {session_id} DELETED ****")
        live_sessions=sessions.keys()
        print(f"*********************** Sessions Live:{live_sessions} ***********************")
        return JSONResponse(content={"message": "Session Deleted", "session_id": session_id},status_code=200) 
        
@app.get("/")
def index():
    return {"message","Server Running"}

@app.post("/upload_files")
async def start(session_id:str=Form(...), files:List[UploadFile]=File(...),model: str = Form(...)):
    try:
        logging.info("**** UPLOADING FILES ****")
        print(model,files,session_id)
        if not supabase.table('sessions').select('session_id').eq('session_id',session_id).execute():
            print(f"**** SESSION {session_id} NOT FOUND****")
            print(e)
            return JSONResponse(content={"error":"Invalid session ID"},status_code=404)
        print("SESSION UPLOAD",session_id)
        print("FILES:",files)
        print("MODEL:",model)
        print("SESSION:",session_id)
        
        if not files or not model:
            return JSONResponse(content={"error":"Kindly Upload Files and Select a model"},status_code=400)
        UPLOAD_DIR=f"pdfs/{session_id}"
        os.makedirs(UPLOAD_DIR,exist_ok=True)
        uploaded_files=[]
        for file in files:
            file_location=os.path.join(UPLOAD_DIR,file.filename)
            with open(file_location,"wb") as f:
                f.write(await file.read())
            uploaded_files.append(file.filename)
        print(f"**** Uploaded Files:{uploaded_files} ****")
        logging.info(f"**** FILES UPLOADED ****")
        logging.info("**** FILES UPLOADED SUCCESSFULLY ****")
        
        supabase.table('sessions').update({
            'model':model,
            'files':uploaded_files
        }).eq('session_id',session_id).execute()
        
        conversation=pipeline(model,session_id)
        sessions[session_id]['conversation']=conversation
        sessions[session_id]['model']=model
        
        logging.info("**** CHAT CREATED, CLEARING PATH ****")
        try:
            shutil.rmtree(UPLOAD_DIR)
            logging.info("**** PATH CLEARED ****")
        except:
            logging.error("**** CANNOT CLEAR PATH ****")
        
        
        #deleting files

        logging.info("**** PATH CLEARED ****")
        logging.info("**** START CHAT ****")
        
        
        return JSONResponse(content={"message": "Files uploaded successfully. Chat Created", "filenames": uploaded_files},status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error":"Files not Uploaded(e). Please upload files again (PDFS ONLY!)"},status_code=400)

@app.post("/chat")
async def chat_start(request: Request):
    data=await request.json()
    
    session_id=data.get('session_id')  
    print(session_id)  
    question=data.get('question')
    
    # session=supabase.table('sessions').select('*').eq('session_id',session_id).execute().data
    if session_id not in sessions or sessions[session_id]['conversation'] is None:
        logging.info("**** SESSION_ID not verified ****")
        return JSONResponse(content={"error": "Conversation model not created yet. Upload files first."}, status_code=400)
    # conversation_json=session[0]['conversation']
    # if not conversation:
    #     logging.info("**** CONVERSATION model not Created ****")
    #     return JSONResponse(content={"error": "Conversation model not created yet. Upload files first."}, status_code=400)
    try:
        #conversation=json.loads(conversation_json)
        response=chat(sessions[session_id]['conversation'],question)
        print(response['answer'])
        logging.info("**** Answered ****")
        return JSONResponse(content={"response":response['answer']})
    except Exception as e:
        logging.info("**** SESSION_ID not verified ****")
        return JSONResponse(content={"error":"Chat Error. Please upload files again"},status_code=400)
        
    
if __name__=="main":
    uvicorn.run(app,port=5001)
    
