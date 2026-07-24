# Low-Light Image Restoration via Retinex-Transformer Architecture

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

Low Light Image Enhancer is a state-of-the-art computer vision pipeline engineered to restore high-quality imagery from extremely low-light environments. This framework implements a hybrid approach, seamlessly uniting physical image decomposition via **Retinex Theory** with deep learning feature restoration using a customized **RetinexFormer** architecture.

---

## 📸 Visual Results

Here is a side-by-side comparison of the low-light input processed through our pipeline versus the ground truth target:

| Input Image (Low-Light) | Enhanced Output (Bright-Light) |
| :---: | :---: |
| ![Dark Input](./data/669.png) | ![Enhanced Output](./data/enhanced.png) |

🚀 **Try the Live Web App:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dark-image-enhancer.streamlit.app/)

---

## 💡 How It Works (Core Concept)

RetinexFormer models an image $I$ based on classical **Retinex Theory**, which decouples an image into pixel-wise components of **Illumination** ($L$, environmental light) and **Reflectance** ($R$, true surface properties):

$$I = L \times R$$




Instead of restoring low-light images end-to-end like a black box, RetinexFormer explicitly estimates the illumination feature map first. This spatial lighting context guides the deep transformer blocks to selectively clean noise and sharpen details in dark regions.
