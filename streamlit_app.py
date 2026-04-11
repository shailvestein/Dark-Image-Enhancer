import streamlit as st
import cv2
import numpy as np
import time
import io
import gc
from PIL import Image
from Models import load_weights
from Enhancer import Enhancer
import torch
import requests

MAX_WIDTH, MAX_HEIGHT = 1920, 1080
MAX_FILE_SIZE = 10 * MAX_WIDTH * MAX_HEIGHT

# --- 1. SET PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="AI image light restoration Lab", page_icon="✨")

# --- 2. SESSION STATE FOR RESET ---
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

def trigger_reset():
    st.session_state.reset_counter += 1
    st.rerun()

# --- 3. Show sample ---
GITHUB_IMAGE_URL = "https://raw.githubusercontent.com/shailvestein/Dark-Image-Enhancer/main/data/ropeway-enh-img.jpg"
response = requests.get(GITHUB_IMAGE_URL)
display_image = Image.open(io.BytesIO(response.content))
st.image(display_image, caption='')

# --- 4. MODEL LOADING ---
@st.cache_resource
def get_enhancer():
    model_fusion = load_weights()
    enhancer = Enhancer(model_fusion, batch_size=2)
    return enhancer

enhancer = get_enhancer()

# --- 5. SIMPLE DOWNLOAD HELPER ---
def get_image_bytes(image_np):
    """Directly converts uint8 numpy array to bytes without extra processing."""
    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    img.save(buf, format='PNG') # PNG is safe and lossless
    return buf.getvalue()

def resize_to_2k(img, target_width=MAX_WIDTH):
    h, w = img.shape[:2]
    if w > target_width:
        aspect_ratio = h / w
        new_width = target_width
        new_height = int(new_width * aspect_ratio)        
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img
    
# --- 6. UI HEADER ---
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>📸 AI Image Light Restoration</h1>", unsafe_allow_html=True)

# --- 7. UPLOADER ---
uploader_key = f"uploader_{st.session_state.reset_counter}"
uploaded_file = st.file_uploader("Upload Low-light Image", type=["jpg", "jpeg", "png"], key=uploader_key)
enhc_img = None

if uploaded_file is not None:
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("❌ File size exceeds.")
    else:
        # Proceed with processing
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_input = cv2.imdecode(file_bytes, 1)
        img_input = resize_to_2k(img_input)
    
        # --- PROCESSING ---
        with st.status("🚀 AI Engine is working...", expanded=True) as status:
            enhc_img, p_time = enhancer.enhance_image(img_input)
            emhc_img = enhc_img * 255
            # enhc_img = cv2.cvtColor(enhc_img, cv2.COLOR_BGR2RGB)
            status.update(label=f"✨ Magic Done in {p_time:.2f}s!", state="complete", expanded=False)
    
        # --- DISPLAY ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='text-align: center;'>🌑 Original</h4>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB), width='stretch')
        with col2:
            st.markdown("<h4 style='text-align: center; color: #00d4ff;'>🌟 Enhanced</h4>", unsafe_allow_html=True)
            st.image(enhc_img, width='stretch')


# --- DOWNLOAD & CLEANUP ---
st.divider()
c1, c2, _ = st.columns([1, 1, 1])

if enhc_img is not None:
    with c1:
        img_bytes = get_image_bytes(enhc_img)
        # Download button click hone par Streamlit refresh hota hai
        if st.download_button("📩 Download Result", data=img_bytes, file_name="enhanced.png", mime="image/png"):
            # User ne download kar liya, ab memory clear karein
            st.success("Download started! Cleaning up server memory...")
            
            # Variables ko delete karein
            if 'img_input' in locals(): del img_input
            if 'enhc_img' in locals(): del enhc_img
            if 'img_bytes' in locals(): del img_bytes
            
            # RAM se force-clear karein
            gc.collect() 
            torch.cuda.empty_cache() # Agar GPU use ho raha hai toh
            trigger_reset()
        
with c2:
    if st.button("🔄 Enhance Another Photo"):
        # Reset session and clear memory
        gc.collect()
        trigger_reset()
    # else:
    #     st.info("👋 Welcome! Please upload a photo to start.")

# --- 8. FOOTER (With Clickable Email) ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; border-top: 1px solid #333; padding-top: 20px;'>
        <p style='color: #888; font-size: 13px; margin-bottom: 5px;'>
            Built with PyTorch & OpenCV
        </p>
        <p style='font-size: 14px;'>
            <span style='color: #555;'>Have a suggestion? </span>
            <a href="mailto:shailvestein.careers@gmail.com?subject=Feedback for DeepSense AI Lab" 
               style="color: #00d4ff; text-decoration: none; font-weight: bold;">
               📩 Contact Developer
            </a>
        </p>
        <p style='color: #00d4ff; font-weight: bold; font-size: 15px; margin-top: 10px;'>
            Powered by Shailesh Vishwakarma
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)
