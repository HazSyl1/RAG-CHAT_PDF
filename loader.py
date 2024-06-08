from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Pinecone 

import os 
import sys
from dotenv import load_dotenv 

import os


loader=PyPDFDirectoryLoader("pdfs")
data=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=20,
    )
text_chunks = text_splitter.split_documents(data)
print(text_chunks[0].page_content)

