"""
Fabric QC Assistant - Streamlit Dashboard
===========================================
This is the final piece: one screen where you upload a fabric photo and see
everything the pipeline produces -- prediction, heatmap, and an AI-generated
QC note grounded in your knowledge base.

SETUP:
    pip install streamlit torch torchvision pytorch-grad-cam sentence-transformers google-generativeai pillow numpy

    Files needed in the same folder as this script:
        - fabric_defect_classifier.pt   (from the training script)
        - qc_knowledge_base.json        (the knowledge base)
        - rag_reasoning_layer.py        (the RAG script, imported below)

    Set your Gemini key before running:
        export GEMINI_API_KEY="your-key-here"

RUN:
    streamlit run streamlit_app.py
"""

import os
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# This imports the RAG logic we already built -- no need to rewrite it here
from rag_reasoning_layer import QCKnowledgeBase, get_qc_note, KB_PATH

# ── Config ──────────────────────────────────────────────────────────────
MODEL_PATH = "fabric_defect_classifier.pt"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.85

# These must be in the EXACT order torchvision's ImageFolder assigned during
# training. ImageFolder sorts class folder names as plain strings, and Python
# sorts capital letters before lowercase ones -- so if your folders are named
# with mixed capitalization (e.g. "Broken stitch" vs "hole"), the order is
# NOT simple alphabetical. Based on your confusion matrix screenshot, your
# folders sorted like this:
CLASS_NAMES = [
    "Broken stitch", "Needle mark", "Pinched fabric", "Vertical",
    "defect free", "hole", "horizontal", "lines", "stain",
]
# DOUBLE-CHECK THIS before running: in your Colab notebook, run
#     print(full_dataset.classes)
# and paste that exact list here. Getting this order wrong means every
# prediction will be labeled with the WRONG class name (the model's math
# would still be correct, just displayed under the wrong name).

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Cached loaders (Streamlit re-runs the whole script on every click,     ─
#    so @st.cache_resource makes sure we only load the heavy stuff ONCE)   ─
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_knowledge_base():
    return QCKnowledgeBase(KB_PATH)


# ── Core pipeline: image in -> everything out ────────────────────────────
def run_pipeline(image: Image.Image, model, kb):
    # Step 1: preprocess the image the same way we did during training
    img_resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized) / 255.0
    input_tensor = eval_transform(img_resized).unsqueeze(0).to(device)

    # Step 2: run the classifier
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        predicted_class = CLASS_NAMES[pred_idx]

    # Step 3: generate the Grad-CAM heatmap
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    heatmap_overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Step 4: ask the RAG layer for a grounded QC note.
    # Note: we lowercase the class name here because qc_knowledge_base.json
    # stores defect types in lowercase (e.g. "hole", "broken stitch"), while
    # the classifier's folder-derived names may be capitalized differently
    # (e.g. "Broken stitch"). We still display the ORIGINAL predicted_class
    # to the user -- only the lookup key is normalized.
    qc_result = get_qc_note(predicted_class.lower(), confidence, kb)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "heatmap_overlay": heatmap_overlay,
        "qc_note": qc_result["note"],
        "needs_human_review": qc_result["needs_human_review"],
    }


# ── Streamlit UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Fabric QC Assistant", layout="wide")
st.title("AI Fabric Quality Assurance Assistant")
st.caption(
    "Upload a fabric image to detect defects, see where the model is looking, "
    "and get an AI-generated QC recommendation grounded in real QC standards."
)

# Keep a running log of low-confidence cases across the session -- this is
# your "human-in-the-loop review queue"
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []

model = load_model()
kb = load_knowledge_base()

uploaded_file = st.file_uploader("Upload a fabric image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Analyzing fabric..."):
        result = run_pipeline(image, model, kb)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM: where the model looked")
        st.image(result["heatmap_overlay"], use_container_width=True)

    st.divider()

    conf_pct = result["confidence"] * 100
    if result["needs_human_review"]:
        st.warning(
            f"**Predicted defect: {result['predicted_class']}** "
            f"({conf_pct:.1f}% confidence -- below the 85% threshold)\n\n"
            "This prediction needs human review before acting on it."
        )
        # Log it into the review queue for the human operator to check later
        st.session_state.review_queue.append({
            "filename": uploaded_file.name,
            "predicted_class": result["predicted_class"],
            "confidence": conf_pct,
        })
    else:
        st.success(f"**Predicted defect: {result['predicted_class']}** ({conf_pct:.1f}% confidence)")

    st.subheader("QC note")
    st.write(result["qc_note"])

# ── Human-in-the-loop review queue (sidebar) ─────────────────────────────
with st.sidebar:
    st.header("Review queue")
    st.caption("Low-confidence predictions land here for manual sign-off.")
    if len(st.session_state.review_queue) == 0:
        st.write("No items pending review.")
    else:
        for i, item in enumerate(st.session_state.review_queue):
            st.write(f"**{item['filename']}** -- {item['predicted_class']} ({item['confidence']:.1f}%)")
            if st.button(f"Mark reviewed", key=f"review_{i}"):
                st.session_state.review_queue.pop(i)
                st.rerun()
