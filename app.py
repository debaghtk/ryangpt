import os
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from dotenv import load_dotenv
from openai import AsyncOpenAI
import tiktoken
import asyncio

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=openai_api_key)

# Define folder path
folder_path = 'transcriptions'

# Load all text files from the folder
documents = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        loader = TextLoader(file_path)
        loaded_documents = loader.load()

        # Extract video ID from filename
        base_name = os.path.splitext(filename)[0]
        video_id = base_name.replace("_transcription", "")
        youtube_link = f"https://www.youtube.com/watch?v={video_id}"

        # Attach metadata with YouTube link information
        for doc in loaded_documents:
            doc.metadata = {"youtube_link": youtube_link}
        documents.extend(loaded_documents)

# Split documents into smaller chunks for easier retrieval
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = []
for document in documents:
    split_texts = text_splitter.split_text(document.page_content)
    for split_text in split_texts:
        split_doc = Document(page_content=split_text, metadata=document.metadata)
        split_documents.append(split_doc)

# Create OpenAI embeddings and use FAISS as the vector store for retrieval
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = FAISS.from_documents(split_documents, embeddings)

# Define the conversation history
conversation_history = []

# Function to count tokens
def count_tokens(messages):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return sum(len(encoding.encode(message['content'])) for message in messages if 'content' in message)

# Function to truncate messages to fit within the token limit
def truncate_messages(messages, max_tokens):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    truncated_messages = []
    total_tokens = 0

    # Try to include the system message if it exists
    system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
    if system_message:
        truncated_messages.append(system_message)
        total_tokens = count_tokens([system_message])

    # Always include the latest user message
    user_message = next(msg for msg in reversed(messages) if msg['role'] == 'user')
    truncated_messages.append(user_message)
    total_tokens += count_tokens([user_message])

    # Add other messages if there's room, prioritizing more recent messages
    for message in reversed(messages):
        if message['role'] != 'system' and message != user_message:
            message_tokens = count_tokens([message])
            if total_tokens + message_tokens <= max_tokens:
                truncated_messages.insert(1, message)  # Insert after system message or at the beginning
                total_tokens += message_tokens
            else:
                break

    return truncated_messages

# Function to get dynamic threshold
def get_dynamic_threshold(scores):
    if not scores:
        return 0
    mean_score = sum(scores) / len(scores)
    return mean_score * 0.8  # You can adjust this multiplier

# Function to handle user input and generate response
async def chatbot_response(user_input, chat_history):
    global conversation_history

    max_context_length = 16000  # Leave some room for the response

    system_message = {"role": "system", "content": "You are Ryan Fernando, a celebrity nutritionist and celebrity himself with more than a million followers on social media. You are answering questions based on the transcription of your YouTube videos. You are also a helpful assistant."}
    
    messages = [system_message] + conversation_history + [{"role": "user", "content": user_input}]

    relevant_docs = vector_store.similarity_search_with_score(user_input, k=3)
    
    # Calculate dynamic threshold
    scores = [score for _, score in relevant_docs]
    threshold = get_dynamic_threshold(scores)

    filtered_docs = []
    for doc, score in relevant_docs:
        if score > threshold:
            filtered_docs.append(doc)
            relevant_text = doc.page_content
            youtube_link = doc.metadata.get("youtube_link", "Link not found")
            messages.append({"role": "system", "content": f"Consider this relevant information (relevance score: {score:.2f}): {relevant_text}"})

        # Check token count and break if we're approaching the limit
        if count_tokens(messages) > max_context_length:
            break

    if filtered_docs:
        messages.append({"role": "system", "content": f"Based on the above information, answer the user's question: {user_input}"})
    else:
        youtube_link = "No relevant video found."

    # Final truncation to ensure we're within limits
    messages = truncate_messages(messages, max_context_length)

    response = await client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    answer = response.choices[0].message.content

    if filtered_docs:
        most_relevant_doc = max(relevant_docs, key=lambda x: x[1])[0]
        most_relevant_link = most_relevant_doc.metadata.get("youtube_link", "Link not found")
        answer_with_link_and_description = f"{answer}\n\nWatch the most relevant video here: {most_relevant_link}"
    else:
        answer_with_link_and_description = f"{answer}\n\nNo relevant video found for this query."

    # Update conversation history, but limit it to maintain context without exceeding token limits
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": answer_with_link_and_description})
    conversation_history = truncate_messages(conversation_history, max_context_length // 2)  # Use half the max length for history

    return answer_with_link_and_description

# Create a Gradio ChatInterface for the chatbot
interface = gr.ChatInterface(
    fn=chatbot_response,
    type="messages"  # Set type to 'messages'
)

# Run the interface
if __name__ == "__main__":
    interface.launch()
