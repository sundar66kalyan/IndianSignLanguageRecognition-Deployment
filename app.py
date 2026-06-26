import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Indian Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🤟 ISL Recognition")

st.sidebar.markdown("""
### Project Information

**Indian Sign Language Recognition**

This application predicts the Indian Sign Language alphabet using a trained MobileNetV2 Deep Learning model.

---

### Supported Classes

A, B, C, D, E, F, G, H, I,
K, L, M, N, O, P, Q,
R, S, T, U, V, W, X, Y

---

### Supported Image Formats

- JPG
- JPEG
- PNG

---

### Model

MobileNetV2

Accuracy: **99%+**

---

Developed by

**KalyanaSundar_AI Engineer**
""")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🤟 Indian Sign Language Recognition")

st.markdown("""
Welcome to the **Indian Sign Language Recognition System**.

This AI-powered application predicts Indian Sign Language alphabet images using a trained **MobileNetV2 Deep Learning model**.

---
""")

# ---------------------------------------------------
# HOW TO USE
# ---------------------------------------------------

st.subheader("📖 How to Use")

st.markdown("""
1. Capture or select a clear hand-sign image.

2. Click **Browse Files**.

3. Upload an image.

4. Wait a few seconds.

5. View the predicted sign and confidence score.
""")

# ---------------------------------------------------
# SAMPLE IMAGES
# ---------------------------------------------------

st.subheader("📝 Sample Signs")

st.info("""
Examples:

• A

• B

• C

• D

...

• Y

(24 Indian Sign Language Classes)
""")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = tf.keras.models.load_model(
    "Models/best_mobilenetv2_model.keras"
)

with open(
    "Models/class_labels.json",
    "r"
) as f:
    class_labels = json.load(f)

index_to_class = {
    value: key
    for key, value in class_labels.items()
}

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

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

    prediction = model.predict(
        image,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    predicted_sign = index_to_class[predicted_index]

    st.success(f"✅ Predicted Sign : {predicted_sign}")

    st.info(f"🎯 Confidence : {confidence:.2f}%")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
"""
### 👨‍💻 Developed By

**KalyanaSundar_AI Engineer**

Indian Sign Language Recognition using Deep Learning

MobileNetV2 • TensorFlow • Streamlit • OpenCV
"""
)