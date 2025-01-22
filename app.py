from flask import Flask, request, jsonify, render_template
from src.helper import  download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import  create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embedding = download_embeddings()

index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', max_tokens=500, temperature=0.3)

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human','{input}')
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET','POST'])
def chat():
    query = request.form['msg']
    response = rag_chain.invoke({'input': query})
    return str(response['answer'])

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)