# DeepSense AI: High-Performance Low-Light Image Restoration via Retinex-Transformer Architectures

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

DeepSense AI ek state-of-the-art computer vision pipeline hai jise low-light environments me capture ki gayi images ko restore karne ke liye design kiya gaya hai. Yeh project **OpenCV (cv2)** image processing aur **RetinexFormer** neural network architecture ke hybrid fusion par kaam karta hai.

---

## 🏗 Model Architecture Specifications

Yeh model image restoration ke liye physical prior (Retinex Theory) aur deep learning transformers ka use karta hai. Iska poora architecture niche diya gaya hai:

### 1. Illumination Estimator
* **Functionality:** Input matrix ($X$) se prakash (Illumination Map) ka estimate lagana.
* **Layers:** Yeh computational speed aur resource efficiency banaye rakhne ke liye **Depthwise Separable Convolutions** (3x3 Depthwise Conv $\rightarrow$ 1x1 Pointwise Conv $\rightarrow$ ReLU) ka use karta hai.

### 2. Illumination-Guided Multi-Head Self-Attention (IG-MSA)
* **Core Logic:** Normal self-attention ke alawa, yeh module estimated Illumination Map ka use attention tokens ke liye 'guidance' matrix ke roop me karta hai.
* **Mathematical Interaction:**
  $$Q_{guided} = Q \times (1 + \text{Illumination Map})$$
  Yeh model ko image ke sabse dark (Severely Degraded) pixel areas par zyada focus (Attention Weight) dene ki permission deta hai.

### 3. Symmetric U-Net Denoiser (IGAB Blocks)
* **Bottleneck Split:** Network encoder aur decoder stage me divided hai jahan har layer par **IGAB (Illumination-Guided Attention Block)** jude hue hain.
* **Skip Connections:** Spatial pixel features ko banaye rakhne ke liye symmetric skip connections ka use kiya gaya hai.

```text
       Input Image Matrix (cv2 Array)
                    │
                    ├───► [ Illumination Estimator ] ───► Illumination Map
                    │                │                            │
                    ▼                ▼                            │
         [ Feature Mapping ] ──► [ Retinex Fusion ]               │ (Attention Guidance)
                                     │                            │
                                     ▼                            ▼
                      [ Symmetric U-Net Denoiser (IGAB Transformer Blocks) ]
                                     │
                                     ▼
                        Residual Output Tensor Map
                                     │
                                     ▼
                           [ Final Residual Addition ]
                                     │
                                     ▼
                        Restored Image Output (cv2 Frame)
