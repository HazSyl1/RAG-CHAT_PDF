


print("INDEX:",index)




question="summarise the methadology"
#print(f"QUESTION: {question} VECTORISED: {embeddings.embed_query(question)}")
# ans=index.query(
#     namespace="real",
#     vector=embeddings.embed_query(question),
#     top_k=2,
#     #include_values=True,
#     include_metadata=True,
# )
#print(f"RESULTS:{[[[x['id']],x['metadata']['text']] for x in ans['matches']]}")

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.getenv("PINECONE_API_KEY"), temperature=0.1)
memory=ConversationBufferMemory(
    memory_key='chat_history',return_messages=True   
)
conversation_chain=ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=index.as_retreiver,
    memory=memory
)
response=conversation_chain({'question':question})
print(response)


# qa=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=)

# prompt_template="""
# Use the following piece of content to answer the question. Pleas provide a detailed response to the question.
# In case the answer is not concise , kindly reply "I'm not sure about that question. Please ask something else."

# {context}

# Question: {question}

# """

# prompt=PromptTemplate(template=prompt_template,input_variables=['context','question'])