from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
import sys
from dotenv import load_dotenv 
from pinecone import Pinecone 
from langchain_pinecone import PineconeVectorStore
#from langchain_community.embeddings import GooglePalmEmbeddings
import os
#from langchain_community.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import logging
logging.basicConfig(level=logging.INFO)
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
#os.environ['HF_TOKEN'] =os.getenv("HF_TOKEN")
load_dotenv(override=True)
#print("HF_TOKEN:",os.getenv("HF_TOKEN"))


def pdf_processing(UPLOAD_DIR):
    try:
        loader=PyPDFDirectoryLoader(UPLOAD_DIR)
        data=loader.load()

        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            )
        text_chunks = text_splitter.split_documents(data)
        return text_chunks
    except:
        return ValueError

def del_vectors(model,session_id):
    if(model=="GOOGLE"):
        index_name='rag-cpdf'
    else:
        index_name='rag-mis'
        
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
    
    index=pc.Index(index_name)
    index_stats=index.describe_index_stats()
    if session_id in index_stats['namespaces'].keys():
        index.delete(delete_all=True,namespace=session_id)
        print("VECTOR FOUND:DELETED")
    else:
        print("NO VECTORSTORE")
        
def vectorise(text_chunks,model,session_id):
    if(model=="GOOGLE"):
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        index_name='rag-cpdf'
    else:
        load_dotenv(override=True)
        embeddings = MistralAIEmbeddings(model = "mistral-embed",api_key=os.getenv("MINSTRAL_AI_API_KEY"))
        index_name='rag-mis'
    
    
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
    
    # index=pc.Index(index_name)
    # index_stats=index.describe_index_stats()
    # if "current" in index_stats['namespaces'].keys():
    #     index.delete(delete_all=True,namespace='current')
    
    #logging.info("**** CLEARED SPACE ****")
        
    
    
    vectorstore = PineconeVectorStore.from_documents(
        text_chunks,
        index_name=index_name,
        embedding=embeddings,
        namespace=session_id
    )
    
    return vectorstore
    #Depricated
    # print(index)
    #index.delete(delete_all=True,namespace='real')
    #pc_info=index.describe_index_stats(namespace="real") 
        
    # for i, t in zip(range(len(text_chunks)), text_chunks):
    #     query_result = embeddings.embed_query(t.page_content)
    #     index.upsert(
    #     vectors=[
    #             {
    #                 "id": str(i),  
    #                 "values": query_result, 
    #                 "metadata": {"text":str(text_chunks[i].page_content)} 
    #             }
    #         ],
    #         namespace="real" 
    #     )


def create_conversation(vectorstore,model):
    try:
        if(model=="GOOGLE"):
            llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        else:
            llm = ChatGroq(temperature=0.3, model_name="mixtral-8x7b-32768")
            
    #     system_message_prompt = SystemMessagePromptTemplate.from_template(
    #     "You have to answer the asked questined , if you dont know the answers, say I apologise I'm now able to answer that.  The context you need  is:{context}"
    # )
    #     human_message_prompt = HumanMessagePromptTemplate.from_template(
    #     "{question}"
    # )
        
    #     messages = [
    #             system_message_prompt,
    #             human_message_prompt
    #     ]
    #     qa_prompt = ChatPromptTemplate.from_messages( messages )
        
        memory=ConversationBufferMemory(
        memory_key='chat_history',return_messages=True)
        conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        )
        return conversation_chain
    except Exception as e:
        return KeyError
def chat(conversation,question):
    response=conversation({'question':question})
    print(response)
    return response
    
