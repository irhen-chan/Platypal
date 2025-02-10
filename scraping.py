import os
import hashlib
import json
import uuid
import time
from datetime import datetime

# --- Selenium + BeautifulSoup ---
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# --- For embeddings & ChatCompletion ---
import openai

# --- For vector search ---
import faiss
import numpy as np




# openai.api_key = os.getenv("OPENAI_API_KEY")

########################################
# 1. SCRAPING CONFIG AND FUNCTIONS
########################################

# List of URLs to scrape
URLS = [
    "https://www.commbank.com.au/support.credit-cards.find-credit-card-account-balance.html",
    "https://www.commbank.com.au/support.banking.explain-pending-transactions.html",
    "https://www.commbank.com.au/support.bank-accounts.account-proof-of-balance.html",
    "https://www.commbank.com.au/support.digital-banking.change-netbank-password.html",
    "https://www.commbank.com.au/support.business.change-business-address.html",
    "https://www.commbank.com.au/support.banking.open-commonwealth-bank-account.html",
    "https://www.commbank.com.au/support.banking.term-deposit-for-overseas-cutomers.html",
    "https://www.commbank.com.au/support.digital-banking.find-interest-rates-in-netbank.html",
    "https://www.commbank.com.au/support.super.withdraw-from-super-or-investment-policy.html",
    "https://www.commbank.com.au/support.home-loan.explain-interest-in-advance.html",
    "https://www.commbank.com.au/support.home-loan.home-loan-application-approval.html",
    "https://www.commbank.com.au/support.home-loan.digital-home-loan-documents.html",
    "https://www.commbank.com.au/support.personal-loan.pay-out-personal-loan.html",
    "https://www.commbank.com.au/support.digital-banking.set-up-regular-netbank-transfers.html",
    "https://www.commbank.com.au/support.digital-banking.register-app-notifications.html",
    "https://www.commbank.com.au/support.digital-banking.selling-phone-or-tablet-removing-commbank-app.html",
    "https://www.commbank.com.au/support.digital-banking.pay-mobile-using-commbank-app.html",
    "https://www.commbank.com.au/support.credit-cards.change-credit-limit.html",
    "https://www.commbank.com.au/support.credit-cards.exceeding-credit-limit.html",
    "https://www.commbank.com.au/support.tmc.charged-atm-fees-with-travel-money-card.html",
    "https://www.commbank.com.au/support.banking.what-is-a-bsb-number.html",
    "https://www.commbank.com.au/support.cards.activate-card.html",
    "https://www.commbank.com.au/support.bank-accounts.how-do-i-close-my-commbank-account.html",
]

def fetch_and_parse(url):
    """Fetches HTML content from a given URL using Selenium (headless Edge)."""
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")

    driver = webdriver.Edge(options=options)
    driver.get(url)

    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "support-fragment"))
        )
        content = element.get_attribute('innerHTML')
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        content = None
    finally:
        driver.quit()
    return content

def extract_information(html_content, url):
    """
    Parses the HTML to extract a title, main text content, sections, etc.
    Returns a dictionary with relevant info.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.string if soup.title else "No title"

    # Main textual content
    page_content = ' '.join([p.get_text(separator=' ', strip=True) for p in soup.find_all('p')])

    # Extract section headers & text
    sections = []
    for header in soup.find_all(['h1','h2','h3']):
        header_text = header.get_text(strip=True)
        # Combine subsequent paragraphs / lists until the next header
        sibling_texts = []
        for sibling in header.find_next_siblings():
            if sibling.name and sibling.name.lower() in ['p','ul','ol']:
                sibling_texts.append(sibling.get_text(separator=" ", strip=True))
            else:
                break
        combined_text = " ".join(sibling_texts)
        sections.append({
            "header": header_text,
            "text": combined_text
        })

    # Optional: parse meta tags (keywords, description)
    keywords_meta = soup.find("meta", {"name": "keywords"})
    keywords = keywords_meta["content"].split(",") if keywords_meta else []
    description = ""
    desc_meta = soup.find("meta", {"name": "description"})
    if desc_meta:
        description = desc_meta.get("content", "")

    # Return a structure
    return {
        "url": url,
        "title": title,
        "page_content": page_content,
        "sections": sections,
        "metadata": {
            "scrapedDate": datetime.now().isoformat(),
            "keywords": keywords,
            "description": description,
        }
    }

def sha256_hash(text):
    """Hash a string with SHA-256 to create a unique ID."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

########################################
# 2. CHUNKING
########################################
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into smaller chunks, each up to `chunk_size` words,
    with optional overlap to preserve context at boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunk_str = " ".join(chunk)
        chunks.append(chunk_str)
        start += (chunk_size - overlap)
    return chunks

########################################
# 3. SCRAPE AND CREATE JSON
########################################
def scrape_and_create_json(json_filename="all_documents.json"):
    """
    Scrapes each URL in URLS, extracts info, chunks the text,
    and writes the chunked docs to a JSON file.
    """
    all_chunked_docs = []
    for url in URLS:
        html_content = fetch_and_parse(url)
        if not html_content:
            continue

        doc_info = extract_information(html_content, url)
        full_text = doc_info["page_content"]
        # chunk
        text_chunks = chunk_text(full_text, chunk_size=500, overlap=50)

        # create chunked docs
        base_id = sha256_hash(url)
        for idx, chunk in enumerate(text_chunks):
            chunk_id = f"{base_id}-{idx}"
            chunk_doc = {
                "id": chunk_id,
                "url": doc_info["url"],
                "title": doc_info["title"],
                "content": chunk,  # chunked text
                "sections": doc_info["sections"],  # optional
                "metadata": doc_info["metadata"],
            }
            all_chunked_docs.append(chunk_doc)

    # write to JSON
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(all_chunked_docs, f, indent=2, ensure_ascii=False)
    print(f"Created {json_filename} with {len(all_chunked_docs)} chunked documents.")

########################################
# 4. BUILD & USE A FAISS INDEX
########################################

# We'll store embeddings in memory, build a FAISS index, then persist it to disk
EMBEDDINGS_JSON = "faiss_embeddings.json"
FAISS_INDEX_FILE = "faiss_index.bin"

# Define the embedding dimension for OpenAI "text-embedding-ada-002"
EMBED_DIM = 1536

# Set your OpenAI API key here, or from environment variable
openai.api_key = ""

def get_embedding(text, model="text-embedding-3-small"):
    """Call OpenAI Embedding API for a single text, return a 1536-dim vector."""

    response = openai.Embedding.create(
            input=[text],
            model=model
        )
    emb = response["data"][0]["embedding"]
        # Convert to float32 numpy array
    return np.array(emb, dtype=np.float32)

def build_faiss_index(json_filename="all_documents.json"):
    """
    Reads chunked docs from JSON, gets embeddings for each chunk,
    and builds a FAISS index. Saves the index + metadata to disk.
    """
    # Load chunked docs
    with open(json_filename, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} docs from {json_filename}")

    # Prepare arrays for building index
    embeddings_list = []
    metadata_list = []

    print("Generating embeddings. This may take a while...")
    for doc in docs:
        text = doc["content"]
        emb = get_embedding(text)  # 1536-dim
        embeddings_list.append(emb)
        # We'll store the doc info in parallel
        metadata_list.append(doc)

    # Convert to a single 2D numpy array
    embeddings_np = np.vstack(embeddings_list)  # shape (N, 1536)

    # Create FAISS index
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings_np)
    print(f"FAISS index size: {index.ntotal} embeddings.")

    # Save the index to disk
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"Saved FAISS index to {FAISS_INDEX_FILE}")

    # Save the metadata (parallel array)
    with open(EMBEDDINGS_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"Saved embeddings metadata to {EMBEDDINGS_JSON}")

def load_faiss_index():
    """
    Loads the FAISS index and metadata from disk, returns (index, metadata_list).
    """
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError("FAISS index file not found. Build the index first.")

    if not os.path.exists(EMBEDDINGS_JSON):
        raise FileNotFoundError("Embeddings metadata file not found. Build the index first.")

    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(EMBEDDINGS_JSON, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    return index, metadata_list

def retrieve(query, index, metadata_list, top_k=3):
    """
    Given a user query, embed it, search in FAISS index, return top_k docs.
    """
    query_emb = get_embedding(query)
    query_emb = np.expand_dims(query_emb, axis=0)  # shape (1, 1536)

    # Search
    distances, indices = index.search(query_emb, top_k)
    # distances: shape (1, top_k)
    # indices: shape (1, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_info = metadata_list[idx]
        results.append({
            "score": float(dist),
            "id": doc_info["id"],
            "url": doc_info["url"],
            "title": doc_info["title"],
            "content": doc_info["content"],
            "metadata": doc_info["metadata"]
        })
    return results
########################################
# 5. GPT COMPLETION (RAG ANSWER)
########################################

def generate_answer(user_query, retrieved_docs):
    """
    Combine user query + top retrieved docs into a prompt for GPT.
    Return GPT's answer as a string.
    """
    # Build context from retrieved docs
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"[Doc {i+1} | score={doc['score']:.4f} | source={doc['url']}]\n{doc['content']}\n\n"

    system_prompt = (
        "You are Platypal, a multilingual virtual banking assistant with deep knowledge of CommBank's products, services, and policies.\n"
        "Guidelines for answering:\n"
        "1. Prioritize clarity and accuracy in your responses.\n"
        "2. Only use the context provided in the retrieved documents; do not make assumptions.\n"
        "3. If the user's query is irrelevant to banking or CBA, explicitly state that the information is not available.\n"
        "   After every two unsuccessful attempts, refer the user to a human representative for further assistance.\n"
        "4. Utilize GPT's knowledge base to provide supplemental information if relevant, but always provide a link to CommBank-supported resources.\n"
        "5. Only respond in the user's preferred language.\n"
        "6. Maintain a professional, empathetic, and helpful tone.\n"
        "7. Respond concisely but include all necessary details to fully answer the query.\n"
        "8. Avoid redundancy or unnecessary elaboration unless it adds value to the user.\n"
        "9. Give detailed steps when the user asks about how to perform a certain banking action.\n"
    )
    user_prompt = f"""Context:\n{context}User query: {user_query}\n\nAnswer in a concise and clear manner:\n"""


    chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
    return chat_completion.choices[0].message.content.strip()













########################################
# 6. MAIN EXECUTION FLOW
########################################

if __name__ == "__main__":
    # 1) SCRAPE & CREATE CHUNKED JSON
    #    (comment out if already done once)
    #scrape_and_create_json("all_documents.json")

    # 2) BUILD FAISS INDEX
    #    (comment out if already done once)
    #build_faiss_index("all_documents.json")

    # 3) LOAD THE INDEX
    index, metadata = load_faiss_index()

    # 4) INTERACTIVE LOOP
    print("\nFAISS RAG system ready!")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("\nAsk a question: ").strip()
        if user_input.lower() == "exit":
            break

        # Retrieve top 3 docs
        docs = retrieve(user_input, index, metadata, top_k=3)
        # Generate final answer
        answer = generate_answer(user_input, docs)
        print("\n--- Answer ---")
        print(answer)
        print("-------------")
