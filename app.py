import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain & related imports
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --------------------------
# API KEYS (environment me set kar)
# --------------------------
from dotenv import load_dotenv
load_dotenv() 
api_key = os.getenv("API_KEY")
print(api_key)


# --------------------------
# STEP 1: Transcript fetch + Vector store banate hi (startup pe)
# --------------------------
video_id = "w_gjS6Zi4JM"  # Example video
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
    transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    transcript = "No captions available"

# Text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Embeddings + VectorStore
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

# LLM + Prompt
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    ANSWER only from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=['context','question']
)

# Chain
def format_docs(retriever_docs):
    context_text = "\n\n".join(doc.page_content for doc in retriever_docs)
    return context_text

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser


# --------------------------
# FASTAPI APP
# --------------------------
app = FastAPI()

# CORS (chrome extension ke liye required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

@app.post("/ask")
def ask_question(msg: Message):
    try:
        answer = main_chain.invoke(msg.text)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
