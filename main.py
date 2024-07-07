from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.messages import HumanMessage
from tqdm import tqdm


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def extract_text_from_pdf(pdf_path, pbar):
    reader = PdfReader(pdf_path)
    text = ""
    step_size = round(75/len(reader.pages), 2) - 0.01
    for page in reader.pages:
        text += page.extract_text()
        pbar.update(step_size)
    return text


def split_text_into_chunks(text,  pbar, chunk_size=1000,):
    chunks = []
    step_size = round(23/len(range(0, len(text), chunk_size)), 2) - 0.01
    for i in range(0, len(text), chunk_size):
        chunks.append(Document(text[i: i + chunk_size]))
        pbar.update(step_size)
    return chunks

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

print("""
Welcome to TextbookGPT! 
Just enter your pdf path, and once the GPT trains on it, you can start chatting with the gpt
Once you are finished chatting, just press 'exit' or 'quit' and we will end the chat for you!
""")

pdf_path = input("Enter the path to the PDF file: ").strip()

total_iterations = 100
pbar = tqdm(total=total_iterations, desc="Reading the book... ", ncols=100, bar_format='{l_bar}{bar}')

# Extract text from the PDF
text = extract_text_from_pdf(pdf_path, pbar)

# Split text into chunks
documents = split_text_into_chunks(text, pbar)


# Store chunks in Chroma vector database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma("pdf_chunks", embeddings)
vectorstore.add_texts([doc.page_content for doc in documents], [doc.metadata for doc in documents])
retriever = vectorstore.as_retriever()

# Declare the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

# Contextualize question based on chat history
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


# Answer question 
qa_system_prompt = """You are an assistant for question-answering tasks over a textbook stored in the context. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer from the context, explicitly mention that and then try answering the questin on your own\
If you still cannot answer, say you don't know the answer
Try to keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# managing chat history
store = {}

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

pbar.update(pbar.total - pbar.n)
pbar.close()
print("Reading complete! Let me know your first question :)\n")

chat_history = []

while True:
    question = input("You		: ").strip()
    if question.lower() in ["exit", "quit"]:
        print(f"TextbookGPT	: Hope I was helpful! Do use me again :)")
        break
    response = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), response["answer"]])    
    print(f"TextbookGPT	: {response['answer']}\n")