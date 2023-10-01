import os , sys
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.text_splitter import RecursiveCharacterTextSplitter #text splitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma

load_dotenv(find_dotenv()) # find .env file and loads its value
docs_folder_path=os.environ["folder_path"]
llm = GooglePalm()
llm.temperature = 0.1
documents = []
for file in os.listdir(docs_folder_path):
    if file.endswith(".docx") or file.endswith(".doc"):
        word_path=docs_folder_path +  "\\" + file
        loader= Docx2txtLoader(word_path)
        documents.extend(loader.load())
    elif file.endswith(".pdf"):
        pdf_path=docs_folder_path +  "\\" + file 
        print(pdf_path)
        loader=PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        txt_path=docs_folder_path +  "\\" + file
        print(txt_path)
        loader=TextLoader(txt_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,separators=[" ", ",", "\n"])
documents = text_splitter.split_documents(documents)

embeddings = GooglePalmEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(),verbose=False,return_source_documents=True)

magenta = "\u001b[35m"
green = "\033[0;32m"
cyan = "\u001b[36m"

chat_history = []
print(f"{magenta}---------------------------------------------------------------------------------")
print('                   Welcome to the document interaction chatbot')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result =  qa_chain(
        {"question": query, "chat_history": chat_history})
    print(f"{cyan}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))














