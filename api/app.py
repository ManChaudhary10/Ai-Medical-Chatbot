import os
import time
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from vercel_blob import download

# --- Load env vars (This is fine at global scope) ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") 

# --- Define Base Directory ---
# Get the absolute path of the project's root directory (one level up from 'api')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Tell Flask where to find the 'templates' folder
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# --- Initialize Flask App ---
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- Define Paths (These are just strings, fine at global scope) ---
TMP_PATH = "/tmp"
DB_FAISS_PATH = os.path.join(TMP_PATH, "db_faiss")
ST_CACHE_PATH = os.path.join(TMP_PATH, "st_models") # Path for sentence-transformer models
FAISS_INDEX_FILE = "index.faiss"
FAISS_PKL_FILE = "index.pkl"

# --- LAZY LOADING GLOBALS ---
# Initialize models and DB as None. They will be loaded on the first request.
embedding_model = None
db = None
llm = None
is_initialized = False # Flag to check if initialization has run

def initialize_app():
    """
    Initializes the models and DB. This runs only once on the first request 
    during a cold start.
    """
    global embedding_model, db, llm, is_initialized
    
    # Check if already initialized in this serverless instance
    if is_initialized:
        return

    print("--- STARTING COLD START INITIALIZATION ---")
    start_time = time.time()

    # --- 1. Download FAISS files from Vercel Blob ---
    print(f"Checking for FAISS files in {DB_FAISS_PATH}...")
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    index_path = os.path.join(DB_FAISS_PATH, FAISS_INDEX_FILE)
    pkl_path = os.path.join(DB_FAISS_PATH, FAISS_PKL_FILE)

    if not os.path.exists(index_path) or not os.path.exists(pkl_path):
        print("FAISS db not found. Downloading from Vercel Blob...")
        try:
            download(pathname=FAISS_INDEX_FILE, destination_path=index_path)
            download(pathname=FAISS_PKL_FILE, destination_path=pkl_path)
            print("✅ FAISS files downloaded.")
        except Exception as e:
            print(f"❌ CRITICAL: Error downloading files from Vercel Blob: {e}")
            raise e # Fail fast if DB can't be downloaded
    else:
        print("✅ FAISS files already exist in /tmp.")

    # --- 2. Initialize Embedding Model ---
    # This will download the model to /tmp/st_models
    print("Loading Embedding Model...")
    os.makedirs(ST_CACHE_PATH, exist_ok=True)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=ST_CACHE_PATH
    )
    print("✅ Embedding Model loaded.")

    # --- 3. Load FAISS Vector Store ---
    print("Loading FAISS Vector Store from /tmp...")
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    print("✅ Vector Store loaded.")

    # --- 4. Initialize Groq LLM ---
    print("Initializing Groq LLM...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0.5,
        max_tokens=512
    )
    print("✅ Groq LLM initialized.")
    
    is_initialized = True
    print(f"--- COLD START COMPLETE in {time.time() - start_time:.2f} seconds ---")


# --- Flask Routes ---
@app.route("/")
def home():
    """Serve the main chatbot UI"""
    # This will now correctly find 'index.html' in the root 'templates' folder
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests from frontend"""
    
    # --- Ensure models are loaded ---
    try:
        # This check will run on every chat request
        # It will only run the full initialize_app() ONE time
        if not is_initialized:
            initialize_app()
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize app: {e}")
        return jsonify({"response": "⚠️ Error: The chatbot failed to start. Please try again later."})
        
    # --- Process the chat ---
    data = request.get_json()
    query = data.get("msg", "").strip()

    if not query:
        return jsonify({"response": "Please ask a question."})

    try:
        retriever = db.as_rioretriever(search_kwargs={'k': 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}"
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return jsonify({"response": answer})

    except Exception as e:
        print(f"❌ Error during chat: {e}")
        return jsonify({"response": "⚠️ An error occurred while generating the answer."})

# --- Main Entrypoint ---
if __name__ == "__main__":
    # This part is only for local development
    # On Vercel, Gunicorn/Uvicorn runs the 'app' object
    print("Starting local development server...")
    initialize_app() # Initialize on local start
    app.run(host="0.0.0.0", port=8080)
