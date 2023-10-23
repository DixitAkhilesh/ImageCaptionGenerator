import streamlit as st
from caption_generator import get_image_input
from gtts import gTTS

st.markdown("<h1 style='text-align: center;'>Image Caption Generator</h1>", unsafe_allow_html=True)

# Upload an image
user_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "gif"])

if user_image:
    # Display the uploaded image
    st.image(user_image, use_column_width=True)

    # Process the image
    result = get_image_input(user_image)

    # Generate caption and audio
    tts = gTTS(text=result, lang='en')
    audio_file = f"./Audios/audio.mp3"
    tts.save(audio_file)

    # Display the caption
    st.write("Image Caption:")
    st.write(result)

    # Offer the audio for download
    st.audio(audio_file, format="audio/mp3")

st.info("Please upload an image to generate a caption.")
