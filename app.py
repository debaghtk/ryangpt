import os
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken  # Make sure to install this library if you haven't already


# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Define folder path
folder_path = 'transcriptions'  # Replace with the folder containing your text files

# Load all text files from the folder
documents = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        loader = TextLoader(file_path)
        loaded_documents = loader.load()

        # Extract video ID from filename (assuming format is 'videoID_transcription.txt')
        base_name = os.path.splitext(filename)[0]
        video_id = base_name.replace("_transcription", "")  # Remove '_transcription' from the base name
        youtube_link = f"https://www.youtube.com/watch?v={video_id}"

        # Attach metadata with YouTube link information
        for doc in loaded_documents:
            doc.metadata = {"youtube_link": youtube_link}
            print(f"Attached metadata for document: {youtube_link}")  # Debug print
        documents.extend(loaded_documents)
# Split documents into smaller chunks for easier retrieval
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = []

for document in documents:
    split_texts = text_splitter.split_text(document.page_content)
    for split_text in split_texts:
        # Ensure metadata from the original document is passed on to the split chunks
        split_doc = Document(page_content=split_text, metadata=document.metadata)
        split_documents.append(split_doc)
        print(f"Split document with metadata: {split_doc.metadata}")  # Debug print

# Create OpenAI embeddings and use FAISS as the vector store for retrieval
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = FAISS.from_documents(split_documents, embeddings)

# Define the conversation history
conversation_history = []

# Function to count tokens
def count_tokens(messages):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Adjust model name as needed
    total_tokens = sum(len(encoding.encode(message['content'])) for message in messages)
    return total_tokens

# Function to truncate messages to fit within the token limit
def truncate_messages(messages, max_tokens):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_tokens = 0
    truncated_messages = []

    for message in reversed(messages):
        message_tokens = len(encoding.encode(message['content']))
        if total_tokens + message_tokens > max_tokens:
            break
        total_tokens += message_tokens
        truncated_messages.append(message)

    return list(reversed(truncated_messages))

# Function to handle user input and generate response
def chatbot_response(user_input):
    global conversation_history

    # Define the maximum context length
    max_context_length = 16385

    # Prepare the messages for the API
    messages = [
        {"role": "system", "content": "You are Ryan Fernando, a celebrity nutritionist and celebrity himself with more than a million followers on social media. You are answering questions based on the transcription of your YouTube videos. You are also a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    # Add conversation history to messages
    messages = conversation_history + messages

    # Truncate messages if they exceed the maximum context length
    if count_tokens(messages) > max_context_length:
        messages = truncate_messages(messages, max_context_length)

    # Generate a response using the LLM with the context
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    answer = response.choices[0].message.content

    # Update answer to include the YouTube link if found
    youtube_link = "Link not found"  # Default if no metadata is found
    if youtube_link != "Link not found":
        answer_with_link = f"{answer}\n\nFor more information, watch the video here: {youtube_link}"
    else:
        answer_with_link = answer

    # Update conversation history
    conversation_history.append((user_input, answer_with_link))
    return answer_with_link

# Create a Gradio interface for the chatbot
def user_interaction(input_text, chat_history):
    response = chatbot_response(input_text)
    chat_history.append((input_text, response))
    return chat_history, chat_history

with gr.Blocks() as demo:
    gr.Markdown("## Ryan GPT")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(interactive=True, lines=1, placeholder="Ask a question based on the transcription...")
    submit_btn = gr.Button("Submit")

    submit_btn.click(user_interaction, [user_input, chatbot], [chatbot, chatbot])


# Launch the Gradio app
demo.launch(debug=True)
