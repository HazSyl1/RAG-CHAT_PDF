from pinecone import Pinecone , ServerlessSpec 
from langchain_community.embeddings import GooglePalmEmbeddings
from dotenv import load_dotenv 
import os
from langchain_community.llms import GooglePalm 
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
load_dotenv()


embeddings=GooglePalmEmbeddings()

pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
index=pc.Index('rag-cpdf')
# upsert data
# for i, t in zip(range(len(text_chunks)), text_chunks):
#    query_result = embeddings.embed_query(t.page_content)
#    index.upsert(
#    vectors=[
#         {
#             "id": str(i),  
#             "values": query_result, 
#             "metadata": {"text":str(text_chunks[i].page_content)} 
#         }
#     ],
#     namespace="real" 
# )



pc_info=index.describe_index_stats(namespace="real") 
print(f"VECTOR DATA INFO:{pc_info}")

question="summarise the methadology"
#print(f"QUESTION: {question} VECTORISED: {embeddings.embed_query(question)}")
ans=index.query(
    namespace="real",
    vector=embeddings.embed_query(question),
    top_k=2,
    #include_values=True,
    include_metadata=True,
)
print(f"RESULTS:{[[[x['id']],x['metadata']['text']] for x in ans['matches']]}")

llm=GooglePalm(temperature=0.1)
qa=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=)

prompt_template="""
Use the following piece of content to answer the question. Pleas provide a detailed response to the question.
In case the answer is not concise , kindly reply "I'm not sure about that question. Please ask something else."

{context}

Question: {question}

"""

prompt=PromptTemplate(template=prompt_template,input_variables=['context','question'])