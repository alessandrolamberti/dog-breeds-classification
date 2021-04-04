import streamlit as st
from inference import make_prediction
from PIL import Image

st.title("Dog breed classification app")
st.text("Upload a dog image for image classification")

uploaded_file = st.file_uploader("Choose a dog photo...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', use_column_width=True)
    st.write("")
    st.write("Oh, isn't he a cute boy")
    st.write("Let's see...")
    label = make_prediction(image)
    st.markdown(f'Uhm, well, the dog is probably a {label[0]}')
    st.markdown(f"P.S: i'm sorry if I got it wrong :(")

        