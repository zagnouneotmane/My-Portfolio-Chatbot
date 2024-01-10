import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, user_template, otmane_template



def get_url_document(URLs):
  
  loaders=UnstructuredURLLoader(urls=URLs)
  url_data=loaders.load()
  return url_data


def get_chunks(document):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(document)
    return chunks

  
def get_vectorstore(document_chunks):

    # embeddings = HuggingFaceInstructEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings()  
  
    vector_store = FAISS.from_documents(document_chunks, embeddings)
  
    return vector_store
def get_llm_model(repo_id):
    # llm = OpenAI()
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.5, "max_length":512})
    
    return llm

def get_conversation_chain(llm, vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
      llm=llm, 
      retriever=vectorstore.as_retriever(), 
      memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(otmane_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Otmane",
                       page_icon=":speech_balloon:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    # get url text
    URLs=[
    'https://zagnouneotmane.github.io/',
    'https://zagnouneotmane.github.io/resume.html',
    'https://zagnouneotmane.github.io/projects.html',
    'https://zagnouneotmane.github.io/certificates.html'


    ]
    raw_document = get_url_document(URLs=URLs)

    # get the text chunks
    document_chunks = get_chunks(raw_document)

    # create vector store
    vectorstore = get_vectorstore(document_chunks)

    # choose llm model
    llm = get_llm_model("google/flan-t5-xxl")
    
    
    

    with st.sidebar:
        st.markdown("### Explore More :mag_right:")
        st.subheader("Unlock the secrets of Otmane's world. Click on 'New Chat' for a journey into the unknown!")
        if st.button("New Chat"):
            with st.spinner("Give it a moment, just wait a few seconds..."):
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(llm, 
                                                                       vectorstore)
                   
                st.success("Let's chat! ")
                



    st.header("Chat with Otmane :speech_balloon:")
    user_question = st.text_input("Have something to ask about me? Go ahead!")
    if user_question:
        handle_userinput(user_question)
    else:
        # create conversation chain
        st.session_state.conversation = get_conversation_chain(llm, 
                                                                       vectorstore)

             


if __name__ == '__main__':
    main()