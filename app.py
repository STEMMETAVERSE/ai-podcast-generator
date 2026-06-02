import streamlit as st
from huggingface_hub import InferenceClient
from gtts import gTTS
import tempfile
import os

# =========================
# CONFIG
# =========================

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

st.set_page_config(
    page_title="AI Podcast Generator",
    layout="centered"
)

st.title("🎙️ AI Podcast Generator")

st.info(
    "Enter a topic and generate a podcast script with optional voice narration."
)

# =========================
# INPUT
# =========================

topic = st.text_input(
    "Enter podcast topic"
)

# =========================
# GENERATE PODCAST
# =========================

if st.button("Generate Podcast"):

    if not topic.strip():

        st.warning(
            "Please enter a topic."
        )

    elif not HF_TOKEN:

        st.error(
            "HF_TOKEN secret is missing."
        )

    else:

        with st.spinner(
            "Generating podcast script..."
        ):

            try:

                prompt = f"""
Create a 2-minute podcast script about:

{topic}

Requirements:
- Conversational tone
- Engaging introduction
- Educational content
- Clear conclusion
- Suitable for students
"""

                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.2-1B-Instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=600,
                    temperature=0.7
                )

                script = response.choices[0].message.content

                st.subheader("📜 Podcast Script")

                st.write(script)

                # =========================
                # TEXT TO SPEECH
                # =========================

                tts = gTTS(script)

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp3"
                ) as fp:

                    tts.save(fp.name)

                    st.subheader("🎧 Podcast Audio")

                    st.audio(fp.name)

            except Exception as e:

                st.error(
                    f"System Response: {e}"
                )
