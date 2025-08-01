# Diabetic Retinopathy Exudate Segmentation 

This branch contains various experiments and architectures for **exudate segmentation** from **fundus retinal images**, with the goal of aiding in **Diabetic Retinopathy (DR)** diagnosis.



## 📌 Problem Statement

Detecting **exudates** (white/yellow lipid deposits) is essential for identifying and staging DR. However, this task is challenging due to:
- Exudates being **small and sparse**
- **Variability in fundus images** (brightness, contrast, source)
- Imbalanced data (mostly background pixels)



## 🧭 Overview

This repository explores two primary approaches for exudate detection:

**🔬 1. Traditional Computer Vision (CV) Pipeline**

Implemented in: Diabetic_Retinopathy_CAD.ipynb

- Top-hat / Bottom-hat filtering for highlighting exudates
- Kirsch filters for detecting blood vessels
- Histogram normalization for detecting the optic disc
- Works entirely on green/grayscale channels
- No learning involved (rule-based detection)


**🤖 2. Deep Learning-Based Segmentation**

Using U-Net and U-Net + ResNet backbones with various input types:
- Green Channel + CLAHE
- RGB
- LAB (L channel enhanced with CLAHE)
- Encoder-decoder approaches with pretrained backbones (ResNet50)

**Models are trained using:**
- Binary Cross Entropy (BCE) loss
- Dice Loss, IoU, and combinations
- Visual and quantitative evaluation (Dice, Precision, Recall, IoU)



## 🧪 Experiments Overview

| Notebook                          | Architecture      | Input Type           | Notes                                                                                                           |
|----------------------------------|-------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------|
| `Diabetic_Retinopathy_CAD.ipynb` | Morphological CV  | Green / Grayscale     | Traditional Computer Vision pipeline using:<br>- Top-hat / bottom-hat for exudates<br>- Kirsch filters for blood vessels<br>- Histogram normalization for optic disc detection |
| `U-net_grey_channel.ipynb`       | U-Net             | Green channel + CLAHE | Enhances contrast using green channel only                                                                     |
| `U-net_RGB_channels.ipynb`       | U-Net             | RGB                   | Uses full RGB input                                                                                            |
| `U-net_LAB.ipynb`                | U-Net             | LAB color space       | CLAHE on L channel, normalized A/B channels                                                                    |
| `Encoder-Decoder_grey_channel.ipynb` | U-Net + ResNet  | Green + CLAHE         | Uses pretrained ResNet encoder                                                                                 |
| `Encoder-Decoder_RGB_channels.ipynb` | U-Net + ResNet  | RGB                   | Best results overall using pretrained weights                                                                  |
| `Encoder-Decoder_LAB.ipynb`      | U-Net + ResNet    | LAB space             | Balanced results and better contrast detection                                                                 |



## 🧠 Key Findings

- **RGB + Pretrained Encoder** performs best numerically.
- **LAB Color Space** helps detect subtle exudates due to better brightness separation.
- **Green channel with CLAHE** enhances contrast but may cause confusion with non-exudate white areas.
- **HDF Loss** introduces positional awareness but is overly sensitive and unstable.



## 🧮 Loss Functions Used

- `BCEWithLogitsLoss`
- `Dice Loss` — to handle class imbalance
- `IoU Loss` — for spatial overlap improvement
- `HausdorffDTLoss` *(removed later)* — penalizes edge shifts but was too harsh



## 🧰 Tools & Libraries

- Python (PyTorch, torchvision, albumentations)
- OpenCV, PIL, NumPy, Matplotlib
- [Segmentation Models PyTorch (smp)](https://github.com/qubvel/segmentation_models.pytorch)



## 📁 Folder Structure

```bash
├── U-net_RGB_channels.ipynb
├── U-net_grey_channel.ipynb
├── U-net_LAB.ipynb
├── Encoder-Decoder_RGB_channels.ipynb
├── Encoder-Decoder_grey_channel.ipynb
├── Encoder-Decoder_LAB.ipynb
├── Diabetic_Retinopathy_CAD.ipynb
├── README.md
└── .gitignore
```


## 📈 Sample Results

- Morphological Computer Vision
<img width="1350" height="743" alt="Screenshot 2025-07-18 193434" src="https://github.com/user-attachments/assets/6f4308a3-7f17-496f-bd3e-8eade63eeba8" />

- Deep Learning Based Segmentation
<img width="740" height="720" alt="Screenshot 2025-07-19 013114" src="https://github.com/user-attachments/assets/6924ef86-5b82-4d09-9d5d-08674efc5bbf" />

<img width="718" height="743" alt="Screenshot 2025-07-19 013050" src="https://github.com/user-attachments/assets/1f719716-13ba-4e44-98e9-80af82e78230" />

<img width="831" height="428" alt="Screenshot 2025-07-19 013037" src="https://github.com/user-attachments/assets/cb8d60f2-14b0-43d6-8c92-27b3bc0133cb" />



## 🧩 Next Steps

- Improve ground truth masks (some may miss true exudates).
- Introduce **attention mechanisms** or **transformer-based encoders**.
- Build an **ensemble** combining multiple input types (e.g., RGB + LAB + green).



## 📣 Acknowledgment

This work is part of our **graduation project** on Computer-Aided Diagnosis for Diabetic Retinopathy — inspired by clinical challenges in ophthalmology.


Feel free to explore the notebooks, test the models, and visualize the predictions! 🔍👁️
