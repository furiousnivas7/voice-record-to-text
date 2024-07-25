import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import noisereduce as nr
from textblob import TextBlob
from googletrans import Translator
import spacy

# Set up Streamlit app
st.title("Speech-to-Text Transcription App")

st.write("Upload an audio file and get the transcription with additional features.")

# File uploader for audio files
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Select language for transcription
    language = st.selectbox("Choose the language of the audio file", ("en-US", "es-ES", "fr-FR"))

    # Load audio file using PyDub
    audio = AudioSegment.from_file(uploaded_file)
    audio.export("temp.wav", format="wav")

    # Noise reduction
    raw_audio_data = audio.get_array_of_samples()
    reduced_noise_audio = nr.reduce_noise(y=raw_audio_data, sr=audio.frame_rate)

    # Save the reduced noise audio
    noise_reduced_audio = AudioSegment(
        reduced_noise_audio.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    noise_reduced_audio.export("temp_reduced_noise.wav", format="wav")

    # Use SpeechRecognition to process the noise-reduced audio
    recognizer = sr.Recognizer()
    with sr.AudioFile("temp_reduced_noise.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=language)

    st.write("Transcription:")
    st.write(text)

    # Sentiment Analysis
    sentiment = TextBlob(text).sentiment
    st.write(f"Sentiment: Polarity={sentiment.polarity}, Subjectivity={sentiment.subjectivity}")

    # Translation
    translator = Translator()
    translated_text = translator.translate(text, dest='es')
    st.write(f"Translated Text (Spanish): {translated_text.text}")

    # Keyword Extraction
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks]
    st.write(f"Keywords: {', '.join(keywords)}")

    # Highlight important sections
    important_keywords = ["important", "note", "highlight"]
    highlighted_text = " ".join([f"**{word}**" if word in important_keywords else word for word in text.split()])
    st.write("Highlighted Text:")
    st.write(highlighted_text, unsafe_allow_html=True)

    # Option to download the transcription
    st.download_button("Download Transcription", text, file_name="transcription.txt")
