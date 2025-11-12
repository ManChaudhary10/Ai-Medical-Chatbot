from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# --- Load environment variables safely and explicitly ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
if "GROQ_API_KEY" in os.environ:
    print("⚠️ Removing old GROQ_API_KEY from system environment...")
    del os.environ["GROQ_API_KEY"]
    load_dotenv(dotenv_path, override=True)
    
print("✅ Loaded .env from:", dotenv_path)
print("✅ GROQ_API_KEY from .env:", os.environ.get("GROQ_API_KEY"))

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Vector Database ---
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --- Initialize Groq LLM ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
    max_tokens=512
)

# --- Flask Routes ---
@app.route("/")
def home():
    """Serve the main chatbot UI"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests from frontend"""
    data = request.get_json()
    query = data.get("msg", "").strip()

    if not query:
        return jsonify({"response": "Please ask a question."})

    try:
        # Retrieve relevant documents
        retriever = db.as_retriever(search_kwargs={'k': 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        # Create the prompt
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

        # Get response from Groq LLM
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return jsonify({"response": answer})

    except Exception as e:
        print("❌ Error during chat:", e)
        return jsonify({"response": "⚠️ An error occurred while generating the answer."})

# --- Main Entrypoint ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
