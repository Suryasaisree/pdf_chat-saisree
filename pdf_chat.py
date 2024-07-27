import os
from dotenv import load_dotenv
import anthropic
import pprint
from halo import Halo
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationSummaryBufferMemory
from langchain.embeddings.base import Embeddings
from langchain.chains import ConversationalRetrievalChain
import numpy as np

load_dotenv()
pp = pprint.PrettyPrinter(indent=4)

# Initialize Anthropic client and LangChain Anthropic model
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_KEY"))
langchain_anthropic = ChatAnthropic(
    model=os.getenv("MODEL_NAME"),
    anthropic_api_key=os.getenv("CLAUDE_KEY")
)

class AnthropicEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        response = self.client.completions.create(
            model="claude-2.0",
            prompt=f"{anthropic.HUMAN_PROMPT} Please generate an embedding for the following text: {text}{anthropic.AI_PROMPT}",
            max_tokens_to_sample=1000,
        )
        # This is a placeholder. Actual embedding generation would require more processing.
        # For now, we'll just return a random vector as an example.
        return np.random.rand(1536)  # 1536 is a common embedding size

def process_pdf(pdf_path):
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"PDF loaded. Number of pages: {len(documents)}")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Text split into {len(texts)} chunks")

    print("Creating embeddings and vector store...")
    embeddings = AnthropicEmbeddings(client)
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store created")

    return vectorstore

def setup_conversational_chain(vectorstore):
    # Set up memory
    memory = ConversationSummaryBufferMemory(
        llm=langchain_anthropic,
        max_token_limit=1000,
        memory_key="chat_history",
        return_messages=True
    )

    # Set up conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=langchain_anthropic,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return chain

def generate_response(chain, user_message):
    spinner = Halo(text='Loading...', spinner='dots')
    spinner.start()

    # Get response from the chain
    response = chain({"question": user_message})

    spinner.stop()

    print("Request:")
    pp.pprint(user_message)
    print("Response:")
    pp.pprint(response['answer'])

    return response['answer']

def main():
    pdf_path = "sairesume.pdf"  # Hardcoded PDF path
    print(f"Starting to process PDF: {pdf_path}")
    print("Checking if file exists...")
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found!")
        return
    
    try:
        vectorstore = process_pdf(pdf_path)
        print("PDF processed successfully.")
        
        chain = setup_conversational_chain(vectorstore)
        print("Conversational chain set up.")

        print(f"PDF '{pdf_path}' processed. You can now start chatting!")

        while True:
            input_text = input("You: ")
            if input_text.lower() == "quit":
                break
            response = generate_response(chain, input_text)
            print(f"Claude: {response}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()