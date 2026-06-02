import streamlit as st
from huggingface_hub import InferenceClient
from gtts import gTTS
import os

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

st.title("🎙️ AI Podcast Generator (HF Hub Powered)")

topic = st.text_input("Enter podcast topic")

if st.button("Generate Podcast"):

    if not topic.strip():
        st.warning("Please enter a topic")
        st.stop()

    prompt = f"""
Create a 2-minute engaging podcast script.

Topic: {topic}

Make it:
- conversational
- engaging
- simple English
"""

    with st.spinner("Generating script..."):

        response = client.text_generation(
            model="google/flan-t5-large",
            prompt=prompt,
            max_new_tokens=300
        )

    st.subheader("📜 Script")
    st.write(response)

    # =========================
    # TEXT TO SPEECH
    # =========================

    try:
        tts = gTTS(response)
        tts.save("podcast.mp3")

        st.subheader("🔊 Audio")
        st.audio("podcast.mp3")

    except Exception as e:
        st.error(f"TTS error: {e}")
