from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
import sys
from dotenv import load_dotenv 
from pinecone import Pinecone 
from langchain_pinecone import PineconeVectorStore
#from langchain_community.embeddings import GooglePalmEmbeddings
from dotenv import load_dotenv 
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

load_dotenv()

def pdf_processing():
    loader=PyPDFDirectoryLoader("pdfs")
    data=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        )
    text_chunks = text_splitter.split_documents(data)
    return text_chunks


def vectorise(text_chunks):
    
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_name='rag-cpdf'
    #Making sure the DB does not contain any previous vectors
    
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
    index=pc.Index(index_name)
    index_stats=index.describe_index_stats()
    if "current" in index_stats['namespaces'].keys():
        index.delete(delete_all=True,namespace='current')
        logging.info("**** CLEARED SPACE ****")
    
    vectorstore = PineconeVectorStore.from_documents(
        text_chunks,
        index_name=index_name,
        embedding=embeddings,
        namespace="current"
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


def create_conversation(vectorstore):
    llm = GoogleGenerativeAI(model="models/text-bison-001", temperature=0.1)
    memory=ConversationBufferMemory(
    memory_key='chat_history',return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory)
    return conversation_chain
    
def chat(conversation,question):
    reponse=conversation({'question':question})
    return reponse
    
