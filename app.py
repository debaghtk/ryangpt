import os
import gradio as gr
from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

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
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(split_documents, embeddings)

# Set up the OpenAI LLM model
llm = OpenAI(temperature=0.3, model="gpt-3.5-turbo")  # Lower temperature keeps responses more fact-based

# Define the conversation history
conversation_history = []

# Function to handle user input and generate response
def chatbot_response(user_input):
    global conversation_history

    # Retrieve relevant documents based on user input
    relevant_docs = vector_store.similarity_search(user_input, k=3)
    if not relevant_docs:
        return "I couldn't find relevant information."

    # Collect the content from relevant documents and prepare a context
    context = ""
    youtube_link = "Link not found"  # Default if no metadata is found
    for doc in relevant_docs:
        context += doc.page_content + "\n"
        if "youtube_link" in doc.metadata:
            youtube_link = doc.metadata["youtube_link"]

    # Generate a response using the LLM with the context
    prompt = f"Context: {context}\n\nQuestion: {user_input}\nAnswer:"
    answer = llm(prompt)

    # Update answer to include the YouTube link
    answer_with_link = f"{answer}\n\nFor more information, watch the video here: {youtube_link}"

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
    user_input = gr.Textbox(placeholder="Ask a question based on the transcription...")
    submit_btn = gr.Button("Submit")

    submit_btn.click(user_interaction, [user_input, chatbot], [chatbot, chatbot])

# Launch the Gradio app
demo.launch()
