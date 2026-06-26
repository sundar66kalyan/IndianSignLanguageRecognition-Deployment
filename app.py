import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Indian Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Indian Sign Language Recognition")
st.write("Upload an Indian Sign Language image for prediction.")

# -----------------------------
# LOAD MODEL
# -----------------------------

model = tf.keras.models.load_model(
    "models/best_mobilenetv2_model.keras"
)

with open(
    "models/class_labels.json",
    "r"
) as f:

    class_labels = json.load(f)

index_to_class = {
    value: key
    for key, value in class_labels.items()
}

# -----------------------------
# IMAGE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(

    "Choose an Image",

    type=["jpg","jpeg","png"]

)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------
    # PREPROCESS
    # -----------------------------

    image = np.array(image)

    image = cv2.resize(
        image,
        (224,224)
    )

    image = image.astype("float32") / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    # -----------------------------
    # PREDICTION
    # -----------------------------

    prediction = model.predict(
        image,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100

    predicted_sign = index_to_class[predicted_index]

    # -----------------------------
    # RESULT
    # -----------------------------

    st.success(
        f"Predicted Sign : {predicted_sign}"
    )

    st.info(
        f"Confidence : {confidence:.2f}%"
    )