import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow import keras

class_dict = {0: 'Covid_19',
              1: 'Normal',
              2: 'Pneumonia'}

st.title("Chest x-ray Covid predictor")

submit = st.button("Example of the chest x-ray image")
if submit:
    image_example = Image.open("../example_lung.jpg")
    st.image(image_example, caption='Example of the chest x-ray image', width=512)

uploaded_file = st.file_uploader("Choose chest x-ray image. Recommended resolution was more then (512, 512).",
                                 type="jpg")

if uploaded_file is not None:
    image_orig = Image.open(uploaded_file).convert('RGB')
    st.image(image_orig, caption='Uploaded image', use_column_width=True)
    st.write("Classifying...")
    model = keras.models.load_model(r"StreamlitLangs_demo/custom_modell.hdf5")
    my_bar = st.progress(0)

    img_array = np.array(image_orig)
    img_array = cv2.resize(img_array, (512, 512), interpolation=cv2.INTER_NEAREST)  # norm
    my_bar.progress(25)
    test_im = np.expand_dims(img_array, axis=0)

    my_bar.progress(50)
    prediction = model.predict(test_im / 255)
    pred = np.argmax(prediction)
    pred_class = class_dict[pred]

    my_bar.progress(100)
    st.success('Done')
    val_round = int(round(np.max(prediction), 2) * 100)
    if val_round > 75:
        st.write(f'**{pred_class}**'
                 f' with the probability **{val_round}%**.')
    else:
        st.write(f"Classificator can't give you a definite answer. The highest score was **{pred_class}**"
                 f" with the probability **{val_round}%**.")
