# --- IMPORTS: The Gathering of Power ---
import os
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # NEW: To serve HTML content
from fastapi.staticfiles import StaticFiles # NEW: To serve static files (JS, CSS)
import uvicorn
import time
import json 

# --- PHASE 1: FORGE THE BRAIN (System Initialization) ---
# This sacred ritual runs only once, at the dawn of the server.

def forge_the_brain():
    """
    Transmutes raw text into a semantic weapon: an in-memory vector store.
    This is not just loading data. This is encoding meaning.
    """
    print("COMMAND: Forging the semantic brain...")
    
    # 1. Procure the Ground Truth
    with open("knowledge_base_en.txt", "r", encoding="utf-8") as f:
        ground_truth = f.read()

    # 2. Shatter into Shards of Knowledge (Chunking)
    shards = ["Q:" + chunk for chunk in ground_truth.split('Q:')[1:]]
    print(f"ANALYSIS: Ground truth shattered into {len(shards)} knowledge shards.")

    # 3. Bestow Meaning upon Shards (Embedding) in Batches
    print("COMMAND: Encoding meaning into shards via embedding model in batches...")
    embedding_model = "models/embedding-001" # Using default for consistency
    
    all_embeddings = []
    batch_size = 5 # Process 5 shards at a time
    for i in range(0, len(shards), batch_size):
        batch = shards[i:i+batch_size]
        print(f"  - Processing batch {i//batch_size + 1}...")
        
        # This is where we call the API for the current batch
        response = genai.embed_content(
            model=embedding_model,
            content=batch,
            task_type="RETRIEVAL_DOCUMENT",
        )
        all_embeddings.extend(response['embedding'])
        
        # Add a small delay to respect API rate limits
        time.sleep(1) # Wait for 1 second before the next batch

    # A matrix of pure meaning, ready for high-speed combat.
    semantic_matrix = np.array(all_embeddings)
    print(f"STATUS: Semantic brain forged with {len(semantic_matrix)} vectors. Awaiting queries.")
    
    return shards, semantic_matrix, embedding_model

# --- EXECUTION: AWAKEN THE MACHINE ---
# Authenticate with the Global Intelligence Network
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise SystemExit("FATAL ERROR: GOOGLE_API_KEY not found. The machine cannot awaken without its key.")

# Forge the brain and keep it online
KNOWLEDGE_SHARDS, SEMANTIC_MATRIX, EMBEDDING_MODEL = forge_the_brain()
# Summon the Oracle (The LLM)
ORACLE = genai.GenerativeModel('gemini-1.5-flash') 

# --- PHASE 2: THE ORACLE'S SANCTUM (FastAPI Application) ---
app = FastAPI(title="The Oracle at Cinematic Dreams")

# NEW: Mount static files (HTML, JS, CSS) from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Grant diplomatic immunity to all origins (CORS)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# NEW: Route to serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the main HTML page from the static directory."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

class Query(BaseModel):
    query: str

@app.post("/ask")
async def consult_the_oracle(query: Query):
    """
    This is not an endpoint. This is the ritual of consultation.
    """
    # 1. Translate the mortal's query into the language of gods (Embed Query)
    query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=query.query, task_type="RETRIEVAL_QUERY")["embedding"]

    # 2. Seek the Lost Knowledge (Semantic Search)
    similarities = np.dot(SEMANTIC_MATRIX, np.array(query_embedding))
    top_k_indices = np.argsort(similarities)[-3:][::-1]
    
    # 3. Assemble the Sacred Context
    context = "\n---\n".join([KNOWLEDGE_SHARDS[i] for i in top_k_indices])

    # 4. Issue the Divine Command (Prompt Engineering)
    command = f"""
    Your designation is 'The Cinematic Dreams Oracle'. Your operational mandate is to assist users based *exclusively* on the provided intelligence context.
    Deviation is forbidden. Fabrication is forbidden. If the context is insufficient, you will state: "That information is beyond my current knowledge."

    INTELLIGENCE CONTEXT:
    ---
    {context}
    ---

    USER'S QUERY:
    "{query.query}"
    
    ORACLE'S RESPONSE:
    """
    
    # 5. Receive the Prophecy (Generation)
    try:
        prophecy = ORACLE.generate_content(command)
        return {"answer": prophecy.text, "debug_context": context} 
    except Exception as e:
        print(f"ORACLE ERROR: {e}")
        raise HTTPException(status_code=503, detail={"message": "The Oracle is currently silent. The connection has been lost.", "debug_info": str(e)})

# ENDPOINT TO GENERATE SUGGESTED QUESTIONS WITH CATEGORIES
@app.get("/suggest-questions")
async def suggest_questions():
    """
    Generates 3 contextually relevant questions and 2 out-of-scope questions for testing,
    ensuring clear categorization.
    """
    print("COMMAND: Generating mixed suggested questions...")
    
    # Predefined list of truly out-of-scope questions for diversity
    all_unrelated_questions = [
        "What's the capital of France?",
        "Can you help me debug my Python code?",
        "What's the weather like today?",
        "Tell me a joke.",
        "How old is the Earth?",
        "What's the best time to invest in stocks?",
        "Who invented the internet?"
    ]
    
    # Choose 2 distinct unrelated questions randomly
    num_unrelated_to_pick = 2
    unrelated_questions_chosen = np.random.choice(
        all_unrelated_questions, 
        min(len(all_unrelated_questions), num_unrelated_to_pick), 
        replace=False
    ).tolist()

    # Generate 3 related questions from knowledge base
    related_questions_list = []
    try:
        num_related_to_pick = 3
        if len(KNOWLEDGE_SHARDS) < num_related_to_pick:
            selected_shards = KNOWLEDGE_SHARDS 
        else:
            random_indices = np.random.choice(len(KNOWLEDGE_SHARDS), num_related_to_pick, replace=False)
            selected_shards = [KNOWLEDGE_SHARDS[i] for i in random_indices]
        
        context_for_related = "\n---\n".join(selected_shards)

        command_related = f"""
        Based on the following Questions and Answers from a cinema's FAQ, generate {len(selected_shards)} alternative ways a user might ask these questions.
        Rephrase them naturally. Do not just copy the original questions. Make them sound like a real person asking.
        Output ONLY a JSON array of {len(selected_shards)} strings. Do NOT include any other text or formatting.

        CONTEXT:
        {context_for_related}

        JSON ARRAY OF {len(selected_shards)} REPHRASED QUESTIONS:
        """
        response_related = ORACLE.generate_content(command_related)
        clean_related_response = response_related.text.strip().replace("```json", "").replace("```", "")
        
        related_questions_list = json.loads(clean_related_response)
        
    except Exception as e:
        print(f"SUGGESTION GENERATION ERROR for related questions: {e}")
        related_questions_list = [
            "Any special discounts on tickets?",
            "What's your refund policy?",
            "Is there parking at the cinema?"
        ]

    return {"related": related_questions_list, "unrelated": unrelated_questions_chosen}

# The __main__ block is removed because the deployment platform will handle running uvicorn.
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
