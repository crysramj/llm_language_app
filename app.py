import os
import streamlit as st
import requests  # Used for making HTTP requests
from dotenv import load_dotenv
from pathlib import Path
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.llms import OpenAI, HuggingFaceHub

# Load environment variables
load_dotenv()

# Load environment variables
PASSWORD = os.getenv('PASSWORD')
XI_API_KEY = os.getenv('ELEVENLABS_API_KEY')
VOICE_ID = "XrExE9yKIg1WjnnlVkGX"  # ID of the voice model to use (currently using Matilda)
OUTPUT_PATH = "output.mp3"  # Path to save the output audio file
CHUNK_SIZE = 1024  # Size of chunks to read/write at a time
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set headers for Streamlit app page
st.set_page_config(page_title="LanguageBot", page_icon=":robot:")
st.header('Welcome to Language - Your AI Language Companion')

def check_password():
    """Returns `True` if the user entered the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    password = st.text_input("Enter password:", type="password")

    if password == PASSWORD:
        st.session_state["password_correct"] = True
        return True
    elif password:
        st.warning("Incorrect password. Please try again.")
    
    return False

if check_password():

    def text_to_speech(text_to_speak: str) -> Path:
        """Convert text to speech and save the audio to a file."""
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

        headers = {
            "Accept": "application/json",
            "xi-api-key": XI_API_KEY
        }

        data = {
            "text": text_to_speak,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0
            }
        }

        response = requests.post(tts_url, headers=headers, json=data, stream=True)

        if response.ok:
            with open(OUTPUT_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
            print("Audio stream saved successfully.")
        else:
            print(response.text)

        return Path(OUTPUT_PATH)

    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize LLM for translation
    template = """
    Sentence: {sentence}, Translation in {language}: Only translate the words specified in the sentence.
    """
    prompt = PromptTemplate(template=template, input_variables=["sentence", "language"])
    llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    language_choice = st.text_input(
        label="**What language would you like to learn?**",
        placeholder="Type in your favorite language!"
    )

    user_sentence = st.text_input(
        label="**Type a sentence?**",
        placeholder="Type in a sentence you would like to learn in this language!"
    )

    if language_choice and user_sentence:
        # Add user sentence to chat history
        st.session_state.messages.append({"role": "user", "content": user_sentence})
        st.chat_message("user").write(f"All text below will be translated to {language_choice}")

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            response = llm_chain.predict(sentence=user_sentence, language=language_choice)
            print(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

            # Generate voice
            audio_file_path = text_to_speech(response)
            if audio_file_path.exists():
                audio_bytes = audio_file_path.read_bytes()
                st.audio(audio_bytes, format='audio/mp3')

    if st.sidebar.button("Reset chat history"):
        st.session_state.messages = []