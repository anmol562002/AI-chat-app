import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader  # Replace PdfFileReader with PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 Panda Chat App powered by Boyanmol')
    st.markdown('''
    ## About
    This app is an babaji powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [boyanmol](https://youtube.com/@boyaanmol)')

def main():
    st.header("Chat with Panda💬")

    load_dotenv()

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:  # Check if a file was uploaded
        st.write(pdf.name)

        # Create a PdfReader object to read the PDF
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            #st.write('Embeddings Loaded from the Disk')
        else:
             embeddings = OpenAIEmbeddings()
             VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
             with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
                #st.write('Embeddings computation completed')
        
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file so panda tell truth it don't lie :")
        #st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
        
            llm = OpenAI(temperature=0,)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
            
            #st.write(docs)

        #st.write(chunks)

        #st.write(text)

if __name__ == '__main__':
    main()
