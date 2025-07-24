# 🩺 **Diabetic Retinopathy Detection** 

This branch focuses on **detecting diabetic retinopathy** from retinal fundus images.  
We also explore the **disease severity levels**, train various deep learning models, visualize predictions using **Grad-CAM**

---

## 🧠 **Project Overview**

- **Diabetic Retinopathy (DR)** is an eye condition caused by diabetes that can lead to vision loss.
- DR is categorized into **five stages**:  
  `0: No DR`, `1: Mild`, `2: Moderate`, `3: Severe`, `4: Proliferative DR`.

This project aims to:
- Classify DR severity from retinal images.
- Improve generalization by training on multiple datasets.
- Explain model decisions with Grad-CAM.
- Prepare a deployable script for real-world use.

---
## 📁 Folder Structure

```plaintext
detection/
│
├── notebooks/
│   ├── dataset_experiments/                  # Experiments on separate datasets
│   │   ├── APTOS_dataset (MobileNet)
│   │   └── DDR_dataset (MobileNet)
│   │
│   ├── combined_dataset_models/              # Training on combined and balanced data
│   │   ├── MobileNet
│   │   ├── EfficientNetB7
│   │   └── Swin Transformer
│   │
│   └── explainability/                       # Grad-CAM visualizations
│       ├── MobileNet model
│       └── Swin Transformer model
│
├── gradcam_utils/                            # Deployment-ready code
│   ├── grad_cam.py                           # Main script for inference with Grad-CAM
│   ├── mobilenet_dr.pth                      # Trained MobileNet weights
│   └── sample.jpg                            # Sample image for testing
│
├── requirements.txt
├── README.md
└── .gitignore
```
---

## 🧪 **Preprocessing Pipeline**
✅ This preprocessing approach is applied across all models used.

   ```Original Fundus Image → Convert to LAB color space → Apply CLAHE on L channel (to enhance contrast) → Merge with A and B channels → Convert back to RGB → Resize to 224×224 → Normalize (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) → Feed into model ```

---

## 📊 **Dataset Experiments**

We started by training a **MobileNet model** separately on:

- ✅ `DDR Dataset`
- ✅ `APTOS Dataset` 

### ⚠ Drawbacks:
- Both datasets suffered from **class imbalance**.
- Some classes (e.g., `Severe`) had too few examples → Overfitting and poor generalization.

---

## 🔗 **Combined Dataset Models**

To mitigate imbalance, we **combined APTOS and DDR datasets**, ensuring better distribution across classes.



### 🧪 Models Applied:
| Model              | Description |
|-------------------|-------------|
| **MobileNetV2**    | Lightweight CNN, ideal for fast inference. |
| **EfficientNetB7** | Deeper and wider variant, achieving state-of-the-art performance. |
| **Swin Transformer** | Vision Transformer using shifted windows for better locality and scalability. |

### 📈 Trials:
Each model notebook contains:
- Training/Validation accuracy plots.
- Trials with loss functions (e.g., CrossEntropy, Focal).
- Regularization (Dropout, LR tuning).
- Scheduler and augmentation trials.
 
---

## 🧠 **Explainability: Grad-CAM Visualizations**

We applied **Grad-CAM** on **test images** using:
- ✅ `MobileNetV2`
- ✅ `Swin Transformer`

### 💡 What is Grad-CAM?
> Grad-CAM (Gradient-weighted Class Activation Mapping) helps visualize **which regions** of an image influenced a model’s prediction.

📍 Used to validate that the model is attending to **disease-relevant areas** (e.g., lesions, hemorrhages).

---

## 🚀 **Deployment Script**

All deployment utilities are in `gradcam_utils/`.

### ✨ Features:
- Loads best-performing MobileNet model.
- Accepts test images.
- Applies preprocessing + Grad-CAM.
- Saves and displays prediction with attention map.

---

## ⚙️ **How to Run**

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run script**:
   ```bash
   python grad_cam.py
   ```
   The script loads `mobilenet_dr.pth` and applies Grad-CAM on `sample.jpg`.

---

## ✅ **Summary**

- ✅ Trained on both individual and combined datasets.
- ✅ Used multiple architectures for benchmarking.
- ✅ Applied Grad-CAM for interpretability.
- ✅ Created reusable script for deployment.

---

📌 *Branch goal: Provide a robust, explainable, and deployable pipeline for diabetic retinopathy detection.*

