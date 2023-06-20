import os
import pandas as pd
import tempfile
import streamlit as st
import fitz
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders.sitemap import SitemapLoader
from langchain import verbose
import pinecone
import asyncio
import nest_asyncio

openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
pinecone_index = st.secrets["PINECONE_INDEX"]
pinecone_namespace = st.secrets["PINECONE_NAMESPACE"]

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
nest_asyncio.apply()

verbose = False


# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
indexes = pinecone.list_indexes()
if st.button(label='Reload Pinecone Indexes', type="primary"):
    indexes = pinecone.list_indexes()



def main():


# Sidebar

    st.sidebar.title("Settings")
    st.sidebar.divider()

    # Create Pinecone Index
    st.sidebar.subheader('Create Index')
    new_index = st.sidebar.text_input('Name your index')
    if st.sidebar.button('Create new index'):
        try:
            if pinecone.create_index(new_index, dimension=1536):
                st.success(f"New index created: {new_index}")
        except pinecone.core.client.exceptions.ApiException as e:
            st.error(f"Failed to create index: {str(e)}")

    # Delete Pinecone Inder
    st.sidebar.subheader('Delete Index')

    delete_select_index = st.sidebar.selectbox("Select a Pinecone index to delete", indexes)

    # Delete Pinecone index
    if st.sidebar.button('Delete PineCone index'):
        if pinecone.delete_index(delete_select_index):
            st.error("Pinecone index deleted")

# End Sidebar


# Main Block
    st.title('CRO Professor')
    
    with st.container():
        st.subheader('Pinecone Index List')
        index_descriptions = []
        vector_counts = []

        try:
            if indexes:
                for index in indexes:
                    desc = pinecone.describe_index(index)
                    index_descriptions.append([desc.name, desc.metric, desc.dimension, desc.metadata_config, desc.status['state']])
                    
                    index_instance = pinecone.Index(index)
                    index_stats_response = index_instance.describe_index_stats()
                    
                    # Compute the total vector count across all namespaces
                    total_vector_count = sum(namespace_info['vector_count'] for namespace_info in index_stats_response['namespaces'].values())
                    
                    vector_counts.append(total_vector_count)

                df = pd.DataFrame(index_descriptions, columns=['Name', 'Metric', 'Dimension', 'Metadata Config', 'Status'])

                df['Vectors'] = vector_counts

                st.dataframe(df)
            else:
                st.error("No indexes were found, please create an index.")
        except pinecone.core.client.exceptions.NotFoundException:
            st.error("No indexes were found, please create an index.")

    st.divider()

    tab1, tab2 = st.tabs(["PDF Uploader", "Sitemap URL Uploader"])

    with tab1:
        # Upload PDF files
        uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True, type="pdf")

        if uploaded_files:     

            with tempfile.TemporaryDirectory() as tmpdir:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    file_content = uploaded_file.read()
                    st.write("Filename: ", file_name)

                    # Write the content of the PDF files to a temporary directory
                    with open(os.path.join(tmpdir, file_name), "wb") as file:
                        file.write(file_content)

                    # Extract and display the PDF metadata using PyPDF2
                    bytes_data = BytesIO(file_content)
                    pdf = PdfReader(bytes_data)
                    metadata = pdf.metadata

                    st.write("PDF Metadata:")
                    for key in metadata:
                        st.write(f"{key}: {metadata[key]}")

                # Load the PDF files from the temporary directory
                loader = DirectoryLoader(tmpdir, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
                documents = loader.load()

                # Split the PDF files into smaller chunks of text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
                documents = text_splitter.split_documents(documents)
                
                # Initialize Pinecone with API keys and environment variables
                pinecone.init(
                    api_key=pinecone_api_key,  # find at app.pinecone.io
                    environment=pinecone_environment  # next to api key in console
                )
                st.success("Connected to Pinecone")
                # Initialize the OpenAI embeddings model
                embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)

                upsert_index_pdf = st.selectbox("Select which index to upload PDF to", indexes)
                
                if st.button(label="Ingest", type="primary"):
                    # Add the documents to Pinecone index
                    Pinecone.from_documents(documents, embeddings, index_name=upsert_index_pdf, namespace=pinecone_namespace)
                    st.success("Ingested File!")

    with tab2:

        # Sitemap URL input
        SitemapURL = st.text_input("Please enter sitemap URL")
        upsert_index_sitemap = st.selectbox("Select which index to upload sitemap to", indexes)
        get_sitemap = st.button(label="Scrape & Ingest", type="primary")
        if get_sitemap:
            try:
                sitemap_loader = SitemapLoader(web_path=SitemapURL)
                st.write('sitemap loader:', sitemap_loader)
                docs = sitemap_loader.load()
                if docs:
                    st.write('Number of Docs Loaded:', len(docs))
                else:
                    st.error('No documents were loaded from the sitemap. Please check the sitemap URL.')
            except Exception as e:
                st.error('An error occurred while loading the sitemap: ' + str(e))


            #Split the docs in smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            st.success('Text Split Done')
            documents = text_splitter.split_documents(docs)
            
            #Initiate PineCone connection
            pinecone.init(
                api_key=pinecone_api_key,  # find at app.pinecone.io
                environment=pinecone_environment  # next to api key in console
                )
            st.success('Pinecone Connected')
            # Initialize the OpenAI embeddings model
            embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
            
            # Add the documents to Pinecone index
            Pinecone.from_documents(documents, embeddings, index_name=upsert_index_sitemap, namespace=pinecone_namespace)
            st.success("Ingested Sitemap!")

if __name__ == '__main__':
    main()
