import streamlit as st
import cv2
import numpy as np
import gc
from Models import load_weights
from Enhancer import Enhancer
import torch
import threading
from streamlit_image_comparison import image_comparison
import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_W, MAX_H = 1920, 1080

if 'lock' not in st.session_state:
    st.session_state.lock = threading.Lock()

st.set_page_config(layout="wide", page_title="AI image light restoration Lab", page_icon="✨")

if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

def trigger_reset():
    st.session_state.reset_counter += 1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

def resize_if_needed(img):
    h, w = img.shape[:2]
    if w > MAX_W or h > MAX_H:
        scale = min(MAX_W / w, MAX_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

try:
    display_image = cv2.imread("./data/ropeway-enh-img.jpg")
    if display_image is not None:
        st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), width='stretch')
except Exception:
    pass

@st.cache_resource
def get_enhancer():
    model = load_weights()
    model.to(device)
    return Enhancer(model, batch_size=2)

def get_image_bytes(image_np):
    success, encoded_img = cv2.imencode('.png', image_np)
    if success:
        return encoded_img.tobytes()
    return b""

st.markdown("<h1 style='text-align: center; color: #00d4ff;'>📸 AI Image Light Restoration Lab</h1>", unsafe_allow_html=True)

uploader_key = f"uploader_{st.session_state.reset_counter}"
uploaded_file = st.file_uploader("Upload Low-light Image (Max 10MB)", type=["jpg", "jpeg", "png"], key=uploader_key)
enhc_img = None

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(f"❌ फाइल बहुत बड़ी है! कृपया 10MB से छोटी इमेज अपलोड करें। (आपकी फाइल: {uploaded_file.size / (1024*1024):.2f} MB)")
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_input = resize_if_needed(img_input)
    
        with st.status("🚀 AI Engine is working...", expanded=True) as status:
            enhancer = get_enhancer()
            with st.session_state.lock:
                enhc_img, p_time = enhancer.enhance_image(img_input)
            status.update(label=f"✨ Magic Done in {p_time:.2f}s!", state="complete", expanded=False)
    
        # यदि मॉडल आउटपुट 0-1 रेंज में है, तो उसे 255 से गुणा करके uint8 में बदलें ताकि वह सही दिखे
        if enhc_img.dtype != np.uint8:
            if np.max(enhc_img) <= 1.0:
                enhc_img = (enhc_img * 255).astype(np.uint8)
            else:
                enhc_img = enhc_img.astype(np.uint8)

        # orig_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        # enhc_rgb = cv2.cvtColor(enhc_img, cv2.COLOR_BGR2RGB)
        
        image_comparison(
            img1=orig_rgb,
            img2=enhc_rgb,
            label1="Original",
            label2="Enhanced",
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )

st.divider()
c1, c2, _ = st.columns([1, 1, 1])

if enhc_img is not None:
    with c1:
        img_bytes = get_image_bytes(enhc_img)
        st.download_button(label="📩 Download Result", data=img_bytes, file_name="enhanced.png", mime="image/png")
        
with c2:
    if st.button("🔄 Enhance Another Photo"):
        trigger_reset()

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; border-top: 1px solid #333; padding-top: 20px;'>
        <p style='color: #888; font-size: 13px; margin-bottom: 5px;'>Built with PyTorch & OpenCV</p>
        <p style='font-size: 14px;'>
            <span style='color: #555;'>Have a suggestion? </span>
            <a href="mailto:shailvestein.careers@gmail.com?subject=Feedback for DeepSense AI Lab" 
               style="color: #00d4ff; text-decoration: none; font-weight: bold;">
               📩 Contact Developer
            </a>
        </p>
        <p style='color: #00d4ff; font-weight: bold; font-size: 15px; margin-top: 10px;'>Powered by Shailesh Vishwakarma</p>
    </div>
    """, 
    unsafe_allow_html=True
)
