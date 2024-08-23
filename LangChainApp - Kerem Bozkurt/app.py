import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests , os
# from IPython.display import Audio

st.set_page_config(page_title="Image To Audio")

load_dotenv(find_dotenv())

def imgToText(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    print(text)
    return text

# url = "https://media-cdn.t24.com.tr/media/stories/2014/08/raw_park-ve-bahcelerde-buyuyen-cocuklar-daha-mutlu_821529041.jpg"
# generated_text = imgToText(url)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def textToSpeech(text):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    print(response)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# audio_data = textToSpeech(generated_text)

# if audio_data:
#     st.audio(audio_data, format="audio/wav")
# else:
#     print("Failed to retrieve audio data.")


def textToImage(text):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.content

image_bytes = textToImage({
	"inputs": "Astronaut riding a horse",
})


def main():
    st.header("Image To Text To Audio To Image")
    uploaded_file = st.file_uploader("Choose an image", type="jpg")
    if uploaded_file is not None:
        # print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        scenario = imgToText(uploaded_file.name)
        audio_data = textToSpeech(scenario)
        with st.expander("Text"):
            st.write(scenario)
        if audio_data:
            st.audio(audio_data, format="audio/wav")
        newImage=textToImage(scenario)
        # print(newImage)
        st.image(newImage, caption="Generated Image", use_column_width=True)



if __name__ == '__main__':
    main()
