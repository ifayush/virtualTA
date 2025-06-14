import json
from typing import List, Dict, Any
import os
from tqdm import tqdm
import requests
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to read and print the contents of the .env file
def read_env_file():
    try:
        with open('.env', 'r') as file:
            print("Contents of .env file:")
            print(file.read())
    except FileNotFoundError:
        print(".env file not found.")

# Call the function to read the .env file
read_env_file()

# Print the AIPROXY_TOKEN for debugging
print("AIPROXY_TOKEN:", os.getenv("AIPROXY_TOKEN"))

# Get API key from environment variable
API_KEY = os.getenv('AIPROXY_TOKEN')
if not API_KEY:
    raise ValueError("AIPROXY_TOKEN environment variable is not set")

# Initialize Pinecone
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Pinecone index
index_name = "discourse-embeddings"
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # OpenAI ada-002 dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

index = pinecone.Index(index_name)

def get_embedding(text):
    """Get embedding for a text using AIproxy."""
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["data"][0]["embedding"]
    else:
        raise Exception(f"Error getting embedding: {response.text}")

def process_posts(filename: str) -> Dict[int, Dict[str, Any]]:
    """Load and group posts by topic"""
    with open(filename, "r", encoding="utf-8") as f:
        posts_data = json.load(f)
    
    topics = {}
    for post in posts_data:
        topic_id = post["topic_id"]
        if topic_id not in topics:
            topics[topic_id] = {
                "topic_title": post.get("topic_title", ""),
                "posts": []
            }
        topics[topic_id]["posts"].append(post)
    
    # Sort posts by post_number
    for topic in topics.values():
        topic["posts"].sort(key=lambda p: p["post_number"])
    
    return topics

def build_thread_map(posts: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Build reply tree structure"""
    thread_map = {}
    for post in posts:
        parent = post.get("reply_to_post_number")
        if parent not in thread_map:
            thread_map[parent] = []
        thread_map[parent].append(post)
    return thread_map

def extract_thread(root_num: int, posts: List[Dict[str, Any]], thread_map: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extract full thread starting from root post"""
    thread = []
    
    def collect_replies(post_num):
        post = next(p for p in posts if p["post_number"] == post_num)
        thread.append(post)
        for reply in thread_map.get(post_num, []):
            collect_replies(reply["post_number"])
    
    collect_replies(root_num)
    return thread

def embed_and_index_threads(topics: Dict[int, Dict[str, Any]], batch_size: int = 100):
    """Embed threads using AIproxy and index in Pinecone"""
    vectors = []
    
    for topic_id, topic_data in tqdm(topics.items()):
        posts = topic_data["posts"]
        topic_title = topic_data["topic_title"]
        thread_map = build_thread_map(posts)
        
        # Process root posts (those without parents)
        root_posts = thread_map.get(None, [])
        for root_post in root_posts:
            thread = extract_thread(root_post["post_number"], posts, thread_map)
            
            # Combine thread text
            combined_text = f"Topic: {topic_title}\n\n"
            combined_text += "\n\n---\n\n".join(
                post["content"].strip() for post in thread
            )
            
            # Get embedding from AIproxy
            embedding = get_embedding(combined_text)
            
            # Prepare vector for Pinecone
            vectors.append({
                "id": f"topic_{topic_id}",
                "values": embedding,
                "metadata": {
                    "topic_id": str(topic_id),
                    "title": topic_title,
                    "post_numbers": [str(p['post_number']) for p in thread],
                    "created_at": thread[0].get('created_at', ''),
                    "last_posted_at": thread[-1].get('created_at', ''),
                    "reply_count": str(len(thread) - 1),
                    "view_count": str(thread[0].get('view_count', 0)),
                    "like_count": str(thread[0].get('like_count', 0))
                }
            })
            
            # Batch upsert when we have enough vectors
            if len(vectors) >= batch_size:
                index.upsert(vectors=vectors)
                vectors = []
    
    # Upsert any remaining vectors
    if vectors:
        index.upsert(vectors=vectors)

def semantic_search(query: str, top_k: int = 3) -> dict:
    """Perform semantic search using Pinecone."""
    try:
        # Get query embedding
        print(f"Getting embedding for query: {query}")
        query_embedding = get_embedding(query)
        
        # Search in Pinecone
        print("Searching in Pinecone...")
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "score": match.score,
                "topic_id": match.metadata["topic_id"],
                "topic_title": match.metadata["title"],
                "post_numbers": match.metadata["post_numbers"],
                "created_at": match.metadata["created_at"],
                "last_posted_at": match.metadata["last_posted_at"],
                "reply_count": match.metadata["reply_count"],
                "view_count": match.metadata["view_count"],
                "like_count": match.metadata["like_count"]
            })
        
        # Generate answer
        print("Generating answer...")
        context_texts = [f"Topic: {res['topic_title']}\nPosts: {res['post_numbers']}" for res in formatted_results]
        answer = generate_answer(query, context_texts)
        
        # Format links
        links = []
        for res in formatted_results:
            links.append({
                "url": f"https://tds.iitm.ac.in/discourse/t/{res['topic_id']}",
                "text": res['topic_title']
            })
        
        return {
            "answer": answer,
            "links": links
        }
        
    except Exception as e:
        print(f"Error in semantic search: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "answer": f"I apologize, but I encountered an error while processing your question. Please try again. Error details: {str(e)}",
            "links": []
        }

def generate_answer(query: str, context_texts: List[str]) -> str:
    """Generate answer using AIproxy"""
    context = "\n\n---\n\n".join(context_texts)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on forum discussions."},
        {"role": "user", "content": f"Based on these forum excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        print(f"Error response from AIproxy: {response.text}")
        raise Exception(f"Error generating answer: {response.text}")

# Example usage
if __name__ == "__main__":
    # Load and process topics
    topics = process_posts("discourse_posts.json")
    print(f"Loaded {len(topics)} topics")
    
    # Embed and index threads
    embed_and_index_threads(topics)
    print("\nIndexing complete\n")
    
    # Test search
    query = "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?"
    print(f"Query: {query}\n")
    
    results = semantic_search(query)
    
    print("Search Results:")
    print(f"Answer: {results['answer']}\n")
    print("Relevant Links:")
    for link in results['links']:
        print(f"- {link['text']}: {link['url']}") 
