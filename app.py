"""
PDF Chat RAG System
Author: Muhammad Ahmad
Description: Chat with your PDFs using embeddings and LLM (RAG + Memory)
"""


import os
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

DATA_DIR='data'

def load_pdfs():
    texts=[]
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.pdf'):
            reader=PdfReader(os.path.join(DATA_DIR,filename))
            full_text=''
            for page in reader.pages:
                full_text+=page.extract_text()+'\n'
            texts.append(full_text)
    return texts

documents=load_pdfs()

def clean_text(text):
    text=re.sub(r"\s+"," ",text)
    return text.strip()

documents=[clean_text(doc) for doc in documents]

def chunk_text(text,chunk_size=3):
    sentences=text.split('. ')
    chunks=[]
    for i in range(0,len(sentences),chunk_size):
        chunk='. '.join(sentences[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

all_chunks=[]
for doc in documents:
    all_chunks.extend(chunk_text(doc))


embed_model=SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings=embed_model.encode(all_chunks)

def search_chunks(query,top_k=3):
    q_emb=embed_model.encode([query])
    similarities=cosine_similarity(q_emb,chunk_embeddings)[0]
    top_idx=similarities.argsort()[-top_k:][::-1]
    return [all_chunks[i] for i in top_idx]

llm=pipeline(
    'text2text-generation',
    model='google/flan-t5-base',
    max_length=250
)


chat_history=[]

def build_memory_context(history,max_turns=3):
    recent=history[-max_turns:]
    text=''
    for q,a in recent:
        text+=f'User: {q}\nAssistant: {a}\n'
    return text


def chat(question):
    memory_text=build_memory_context(chat_history)

    combined_query=memory_text+" "+question
    retrieved=search_chunks(combined_query)
    
    context=''.join(retrieved)

    prompt=f"""
You are an assistant answering questions using the document context and chat history.

Chat History:
{memory_text}

Context:
{context}

Question:
{question}
"""
    
    answer=llm(prompt)[0]['generated_text']
    print('Assistant Answer:',answer)

    chat_history.append((question,answer))


print("PDF CHAT RAG SYSTEM READY")
print("Type 'exit' to stop.\n")

while True:
    q=input("You: ")
    if q.lower()=='exit':
        break
    chat(q)