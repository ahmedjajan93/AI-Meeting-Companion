import streamlit as st 
import torch
import librosa
import whisper
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="🎥 AI Meeting Companion", page_icon=":robot_face:", layout="wide")
st.title("🎥 AI Meeting Companion")

# Initialize Hugging Face LLM (e.g., Zephyr-7B)
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=os.getenv("API_TOKEN"),
)



# Load the Whisper model
@st.cache_resource
def load_whisper_model(model_name):
    return whisper.load_model(model_name)

model = load_whisper_model('medium')

st.write("Upload your video or audio file")
uploaded_file = st.file_uploader("Choose a file", type=["mp4", "mov", "mp3", "wav"])

if uploaded_file is not None:
    with st.spinner("Processing your file..."):
        try:
            
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    temp_audio.write(uploaded_file.read())
                    temp_audio_path = temp_audio.name
                # Directly load the audio file
            audio, sr = librosa.load(temp_audio_path, sr=None)

            # Transcribe audio using Whisper
            result = model.transcribe(audio)
            transcript = result["text"]
            st.write("Transcription successful!")

            # Split transcript into chunks for embedding
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(transcript)

            # Create embeddings and vector store using HuggingFace embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_texts(texts, embeddings) 
            
            # Create retriever
            retriever = db.as_retriever(search_kwargs={"k": 3})

            # Build QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
            )

            # Ask a question
            question = st.text_area("Ask a question related to the meeting transcript:")
            if question:
                with st.spinner("Generating answer..."):
                    answer = qa_chain.run(question)
                st.write("Answer:", answer)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload a video or audio file to proceed.")
