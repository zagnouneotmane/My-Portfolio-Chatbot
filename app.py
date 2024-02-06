import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import CTransformers



def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = FAISS.from_documents(document_chunks, HuggingFaceEmbeddings() )
    
    return vector_store

def get_llm_model():
    # llm = OpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":512})
    #llm = CTransformers(model="TheBloke/Llama-2-7b-Chat-GGML", model_file='llama-2-7b-chat.ggmlv3.q2_K.bin')
    
    return llm

def get_context_retriever_chain(llm, vector_store):
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(llm, retriever_chain): 
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    conversational_rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
    return conversational_rag_chain

def get_response(user_query):
    # get url text
    URLs=[
    'https://zagnouneotmane.github.io/',
    'https://zagnouneotmane.github.io/resume.html',
    #'https://zagnouneotmane.github.io/projects.html',
    #'https://zagnouneotmane.github.io/certificates.html'


    ]
    vector_store = get_vectorstore_from_url(URLs)  
    llm = get_llm_model()
    retriever_chain = get_context_retriever_chain(llm, vector_store)
    conversation_rag_chain = get_conversational_rag_chain(llm, retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']

def main():

    load_dotenv()

    # app config
    st.set_page_config(page_title="Chat with websites", page_icon="🤖")
    st.header("Chat with Otmane :speech_balloon:")

        
    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)    
    else:
    # session state
        st.session_state.chat_history = [
            AIMessage(content="Hello, Have something to ask about me? Go ahead!"),
        ] 

    
    # sidebar
    with st.sidebar:
            st.markdown("### Explore More :mag_right:")
            st.subheader("Unlock the secrets of Otmane's world. Click on 'New Chat' for a journey into the unknown!")
            if st.button("New Chat"):
                with st.spinner("Give it a moment, just wait a few seconds..."):
                    # session state
                    st.session_state.chat_history = [
                        AIMessage(content="Hello, Have something to ask about me? Go ahead!"),
                    ]
                    
                    st.success("Let's chat! ")

if __name__ == '__main__':
    main() 