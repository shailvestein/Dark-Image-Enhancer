# High-Performance Low-Light Image Restoration via Retinex-Transformer Architectures

## 🖼 Visual Results

Here is a side-by-side comparison of the low-light input processed through our pipeline versus the ground truth:

| Input Image (Low-Light) | DeepSense AI Enhanced Output |
| :---: | :---: |
| ![Dark Input](documents/sample_dark.png) | ![Enhanced Output](enhanced_image.png) |

Visit: [![Streamlit CloudApp]()](https://dark-image-enhancer.streamlit.app/)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

Low light image enhancer is a state-of-the-art computer vision pipeline engineered to restore high-quality imagery from extremely low-light environments. This framework implements a hybrid approach, seamlessly uniting robust **OpenCV (cv2)** image preprocessing workflows with a customized **RetinexFormer** neural network architecture.

---

## 🏗 Model Architecture Specifications

The framework incorporates physical priors based on Retinex Theory paired with deep learning transformer architectures for pixel-level feature restoration. The complete system design is detailed below:

### 1. Illumination Estimator
* **Functionality:** Estimates the light illumination map directly from the input frame matrix ($X$).
* **Layers:** Employs **Depthwise Separable Convolutions** (3x3 Depthwise Conv $\rightarrow$ 1x1 Pointwise Conv $\rightarrow$ ReLU) to minimize hardware parameters and guarantee low-latency inference speeds.

### 2. Illumination-Guided Multi-Head Self-Attention (IG-MSA)
* **Core Logic:** Unlike conventional self-attention mechanisms, this module dynamically injects the estimated Illumination Map as a guidance matrix for calculating attention weights.
* **Mathematical Interaction:**
  $$Q_{guided} = Q \times (1 + \text{Illumination Map})$$
  This formulation explicitly forces the transformer layers to assign higher attention weights to severely degraded and dark pixel regions.

### 3. Symmetric U-Net Denoiser (IGAB Blocks)
* **Bottleneck Split:** The network follows an encoder-decoder design pattern where each hierarchical layer is integrated with an **IGAB (Illumination-Guided Attention Block)**.
* **Skip Connections:** Implements symmetric skip connections to preserve high-frequency spatial features and localized pixel context across deep layers.

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
```

### 4. Loss Formulations
The network is optimized using a joint loss function to guarantee both structural fidelity and exposure naturalness:
$$\mathcal{Loss} = \mathcal{L}_{1}

📊 Performance Benchmarks & Evaluation

The model has been rigorously evaluated using structural and peak signal performance targets across standard benchmarks and mixed real-world distributions:
```
Evaluation Dataset Split                  Target Performance (PSNR)
LOL (Low-Light) Dataset                   21.51 dB
Custom Augmented Dataset                  19.00 dB

```


## 🗺 Future Roadmap
* [ ] Integrate TensorRT execution providers for real-time edge streaming inference.
* [ ] Implement zero-shot self-supervised training modules to eliminate reliance on paired datasets.
