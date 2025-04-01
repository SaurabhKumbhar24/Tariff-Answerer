import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time


if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="deepseek-r1")
    st.session_state.vectors = FAISS.load_local("vector_storage",embeddings=st.session_state.embeddings, allow_dangerous_deserialization=True)

st.title("Tariff Answerer.")

llm = Ollama(model="deepseek-r1")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context.
    Provide detailed answer.

    <context>
    {context}
    <context>

    Questions:{input}
    """
)

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    answers = response["answer"].split("</think>")
    st.write(answers[1])

    with st.expander("Thinking Process: "):
        st.write(answers[0][6:])

    with st.expander("Document Similarity Search: "):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)

    st.write("Response Time: ",time.process_time() - start_time)