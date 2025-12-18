import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas  # type: ignore[import]
import time

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="MNIST Digit Classifier",
    layout="wide",
)

# -----------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------
st.sidebar.title("⚙️ Settings")

dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
stroke_size = st.sidebar.slider("🖌️ Brush Size", 5, 30, 15)
canvas_res = st.sidebar.selectbox("🖼️ Canvas Resolution", [200, 300, 400], index=1)
show_heatmap = st.sidebar.checkbox("🔥 Show Confidence Heatmap", value=True)
use_webcam = st.sidebar.checkbox("📷 Use Webcam (optional)", value=False)

# DARK MODE CSS
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #e0e0e0; }
        .stButton>button { background-color: #303030 !important; color: white !important; border-radius: 8px; }
        .stMetric { background-color: #1a1a1a !important; border-radius: 8px; padding: 10px; }
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

# Session state for keeping a small prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []


# -----------------------------------------
# TITLE
# -----------------------------------------
st.title("🧠 MNIST Digit Classifier — Enhanced Edition")
st.write("Draw, upload, or capture a digit to classify it instantly.")
with st.expander("ℹ️ How to use this app", expanded=True):
    st.markdown(
        """
        - **Draw Digit**: Use your mouse/finger to draw a digit (0–9) on the canvas.
        - **Upload Image**: Upload a clear, centered grayscale/handwritten digit.
        - **Webcam Capture**: Enable webcam in the sidebar and hold a digit up to the camera.
        - **Tip**: High‑contrast, centered digits give the best results.
        """
    )


# -----------------------------------------
# TABS
# -----------------------------------------
tab1, tab2, tab3 = st.tabs(["✏️ Draw Digit", "📤 Upload Image", "📷 Webcam Capture"])



# ===================================================
# TAB 1 — DRAW DIGIT
# ===================================================
with tab1:

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("✏️ Draw a digit")

        # Clear canvas button
        if st.button("🧹 Clear Canvas"):
            st.experimental_rerun()

        canvas = st_canvas(
            fill_color="white",
            stroke_width=stroke_size,
            stroke_color="black",
            background_color="white",
            width=canvas_res,
            height=canvas_res,
            drawing_mode="freedraw",
            key="canvas_draw",
        )

    with right:
        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")

            # Detect blank canvas
            if np.mean(canvas.image_data) > 250:
                st.warning("✋ Canvas is blank. Draw something!")
            else:
                img = img.resize((28, 28))
                img = ImageOps.invert(img)
                arr = np.array(img) / 255.0
                arr = arr.reshape(1, 28, 28, 1)

                pred = model.predict(arr)
                pred_label = np.argmax(pred)

                st.metric("Predicted Digit", int(pred_label))

                # Save to history
                st.session_state["history"].append(
                    {
                        "Source": "Draw",
                        "Predicted Digit": int(pred_label),
                        "Confidence": float(pred[0][pred_label]),
                    }
                )

                # Animated confidence bars
                st.write("### 📊 Confidence Level")
                for i in range(10):
                    st.progress(float(pred[0][i]))

                # Top‑3 predictions table
                top3_idx = np.argsort(pred[0])[::-1][:3]
                st.write("### 🏆 Top 3 Predictions")
                st.table(
                    {
                        "Digit": [int(i) for i in top3_idx],
                        "Confidence": [float(pred[0][i]) for i in top3_idx],
                    }
                )

                if show_heatmap:
                    st.write("### 🔥 Confidence Scores")
                    st.json({str(i): float(pred[0][i]) for i in range(10)})



# ===================================================
# TAB 2 — UPLOAD IMAGE
# ===================================================
with tab2:
    st.subheader("📤 Upload a digit image")

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("L")
        st.image(img, width=200, caption="Uploaded Image")

        img28 = img.resize((28, 28))
        arr = np.array(img28) / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        pred = model.predict(arr)
        pred_label = np.argmax(pred)

        st.metric("Predicted Digit", int(pred_label))

        # Save to history
        st.session_state["history"].append(
            {
                "Source": "Upload",
                "Predicted Digit": int(pred_label),
                "Confidence": float(pred[0][pred_label]),
            }
        )

        st.write("### 📊 Confidence Scores")
        st.json({str(i): float(pred[0][i]) for i in range(10)})



# ===================================================
# TAB 3 — WEBCAM INPUT (Optional)
# ===================================================
with tab3:
    st.subheader("📷 Capture Digit From Webcam")

    if use_webcam:
        camera_img = st.camera_input("Take a picture")

        if camera_img:
            img = Image.open(camera_img).convert("L")
            st.image(img, width=200)

            img28 = img.resize((28, 28))
            arr = np.array(img28) / 255.0
            arr = arr.reshape(1, 28, 28, 1)

            pred = model.predict(arr)
            pred_label = np.argmax(pred)

            st.metric("Predicted Digit", int(pred_label))

            # Save to history
            st.session_state["history"].append(
                {
                    "Source": "Webcam",
                    "Predicted Digit": int(pred_label),
                    "Confidence": float(pred[0][pred_label]),
                }
            )

            st.write("### 📊 Confidence Scores")
            st.json({str(i): float(pred[0][i]) for i in range(10)})
    else:
        st.info("Enable *Use Webcam* in the sidebar to activate this feature.")


# ===================================================
# PREDICTION HISTORY
# ===================================================
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("📜 Prediction History (this session)")
    st.dataframe(st.session_state["history"])
