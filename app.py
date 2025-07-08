# --- IMPORTS: The Gathering of Power ---
import os
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # Still useful for internal data validation
from fastapi.middleware.cors import CORSMiddleware # Still useful if calling FastAPI directly later
import uvicorn
import time
import json 
import gradio as gr # <<< NEW: Import Gradio

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


# --- PHASE 2: THE ORACLE'S SANCTUM (FastAPI Application removed for direct Gradio integration) ---
# The FastAPI app instance and its routes (/, /ask, /suggest-questions) are now integrated
# directly into Gradio's functional approach.
# This means:
# - app = FastAPI(...) is no longer needed here.
# - app.mount and @app.get("/") are no longer needed.
# - The @app.post("/ask") logic will be wrapped in a Gradio function.
# - The @app.get("/suggest-questions") logic will also be wrapped in a Gradio function.

# --- GRADIO INTERFACE: THE NEW FRONTEND COMMAND CENTER ---

def consult_the_oracle_gradio(query: str):
    """
    Handles a user's question via Gradio.
    This function wraps the core RAG logic.
    """
    if not query:
        return "Please enter a question.", "" # Returns answer, debug_context

    # 1. Embed the user's query
    query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=query, task_type="RETRIEVAL_QUERY")["embedding"]

    # 2. Seek the Lost Knowledge (Semantic Search)
    similarities = np.dot(SEMANTIC_MATRIX, np.array(query_embedding))
    top_k_indices = np.argsort(similarities)[-3:][::-1] # Get top 3
    
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

    USER'S QUESTION:
    "{query}"
    
    ORACLE'S RESPONSE:
    """
    
    # 5. Receive the Prophecy (Generation)
    try:
        prophecy = ORACLE.generate_content(command)
        return prophecy.text, context # Return both answer and debug_context
    except Exception as e:
        error_msg = f"The Oracle is currently silent. The connection has been lost. Error: {str(e)}"
        print(f"ORACLE ERROR: {e}")
        return error_msg, context # Return error message and context for debugging

def generate_suggestions_gradio():
    """
    Generates 3 contextually relevant questions and 2 out-of-scope questions for testing,
    ensuring clear categorization.
    """
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

    # Gradio expects string outputs for Markdown components.
    # We'll format it as Markdown list
    markdown_output = "#### Related Questions:\n" + "\n".join([f"- {q}" for q in related_questions_list])
    markdown_output += "\n\n#### Out-of-Scope Questions:\n" + "\n".join([f"- {q}" for q in unrelated_questions_chosen])
    
    return markdown_output


# --- GRADIO UI DEFINITION ---
# This is where we define the visual interface for Hugging Face Spaces.

# Chatbot interface
chat_interface = gr.ChatInterface(
    fn=consult_the_oracle_gradio,
    chatbot=gr.Chatbot(height=400, render_markdown=True, label="Cinematic Dreams AI Assistant"),
    textbox=gr.Textbox(placeholder="Ask me anything...", container=False, scale=7),
    title="Cinematic Dreams AI Assistant",
    description="Your intelligent guide to all things cinema. Powered by Google Gemini and RAG.",
    theme=gr.themes.Monochrome(), # Using a clean, modern theme
    examples=[
        "What movies are currently playing?",
        "Can I bring food from outside?",
        "Are there any student discounts?",
        "What's the capital of France?" 
    ], # Simple examples, but not dynamic like the suggestions endpoint
    undo_btn=None,
    clear_btn="Clear Chat",
)

# Debug and Suggestion tab
with gr.Blocks(theme=gr.themes.Monochrome()) as debug_tab:
    gr.Markdown("### Debug Information & Test Suggestions")
    
    # Debug context display
    debug_context_output = gr.Textbox(label="Last Retrieved Context (for Debugging)", lines=10, interactive=False, visible=True)
    
    # Integrate consult_the_oracle_gradio to also update debug_context_output
    # This needs to be done carefully with state and separate blocks/interfaces.
    # For simplicity, we'll connect the main chat output to debug context,
    # or rely on the chat history itself.

    gr.Markdown("### Test Questions")
    suggestion_output = gr.Markdown(value="Generating suggestions...", visible=True)
    generate_btn = gr.Button("Generate New Test Questions")
    
    generate_btn.click(
        fn=generate_suggestions_gradio,
        inputs=None,
        outputs=suggestion_output
    )
    
    # We will dynamically update the debug context from the chat_interface
    # This requires more complex Gradio state management, beyond a simple
    # functional example for initial deployment.
    # For now, the `debug_context` will be printed to logs and you can check it there.
    # In a full Gradio app, you'd chain the output of `consult_the_oracle_gradio`
    # to update this `debug_context_output` component.

# Create the main Gradio interface with tabs
demo = gr.TabbedInterface(
    [chat_interface, debug_tab],
    ["Chat", "Debug / Suggestions"]
)

# This is the entry point for Gradio.
# Hugging Face Spaces will automatically run this.
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
