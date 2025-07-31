# 👁️ Diabetic Retinopathy CAD System

## 📋 Table of Contents
* [📌 Introduction](#1-introduction)  
* [🔍 Project Overview](#2-project-overview)
    * [🎯 Disease Grading](#2.1-Disease-Grading)  
    * [🧠 Exudate Segmentation](#2.2-Exudate-Segmentation)  
    * [💬 Interactive Chatbot](#2.3-Interactive-Chatbot)  
* [📁 Project Structure](#3-Project-Structure)  
* [💡Key Technologies](#4-key-technologies)  
* [🚀 How to Run the Application](#5-how-to-run-the-application)  
* [🔮 Future Work](#6-future-work)  
* [🤝 Let's Collaborate](#7-lets-collaborate)  
* [👥 Collaborators](#8-Collaborators)  

---

## 1. **Introduction**  
Welcome to the **Diabetic Retinopathy Classification Project—a Streamlit-based web application** that enables:
- **AI-Powered Analysis**: Utilizes deep learning models for accurate detection of diabetic retinopathy
- **Multi-Language Support**: Available in English, Arabic, French, and Spanish
- **Interactive Chatbot**: AI assistant to answer questions about diabetic retinopathy
- **Detailed Reports**: Generates comprehensive PDF reports with analysis results
- **User-Friendly Interface**: Intuitive and accessible design for all users


This tool supports both healthcare professionals and patients, making early diagnosis and education more accessible through explainable AI and a friendly interface.

---

## 2. **Project Overview**

This project is divided into *three main components*, each developed and maintained in a dedicated branch:

### 2.1 **Disease Grading**  
We trained multiple models to *predict the grade of diabetic retinopathy* from fundus images. The grades are:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

#### ✅ Models used:
- *MobileNetV2*  
- *EfficientNet-B0 and EfficientNet-B7*  
- *Swin Transformer*

After experiments, *MobileNetV2* was selected for deployment based on its balance of performance and speed. We also integrated *Grad-CAM* to visualize which parts of the retina the model used for its decisions.  
All experiments and results are available in the notebooks under the **detection branch**.



### 2.2 **Exudate Segmentation**  
This part aims to *segment exudates*—bright lesions in the retina that appear due to leakage from blood vessels and are key signs of diabetic retinopathy progression.

#### 🔍 Why Exudates?  
Exudates are among the earliest clinical signs of DR. Accurate segmentation helps:
- Understand lesion load,
- Guide diagnosis/treatment,
- Enable fine-grained analysis for early detection.

We used a segmentation architecture trained on expert-annotated data. Notebooks and scripts can be found in the **segmentation branch**.



### 2.3 **Interactive Chatbot**  
This module features a *RAG-based (Retrieval-Augmented Generation) chatbot* that answers *medical questions* about diabetic retinopathy using reliable medical sources.

#### 🤖 Capabilities:
- Understands queries about symptoms, causes, treatments, grading systems, etc.
- Retrieves relevant info from documents and generates answers.
- Built to assist *clinicians and patients* in gaining instant access to trustworthy knowledge.

You can find the chatbot pipeline and Streamlit app in the **chatbot branch**.

---


## 3. **Project Structure**

```
 📁 app/                             # Streamlit application
│   ├── app1.py
│   ├── style.css
│   ├── report_generator.py
│
├── 📁 chatbot/                         # Chatbot components
│   ├── chatbot.py
│   ├── build_knowledge.py
│   ├── system_prompt.txt
│   ├── 📁 vector_db/                   # Vector DB for RAG
│
├── 📁 models/                          # Saved models
│   └── mobilenet_dr(70%).pth
│   └── rgb_model.pth
│
├── 📁 explainability/                 # Grad-CAM scripts and outputs
│   ├── grad_cam.py
│   ├── gradcam_dr.jpg
│   ├── gradcam_temp.jpg
│
├── 📁 segmentation/                   # Exudate segmentation
│   ├── exudates_inference.py
│   ├── mask_exudates.jpg
│   ├── mask_ex.jpg
│
├── 📁 data/                           
│   ├── jciinsight-2-93751.pdf
│   ├── 978-981-10-3509-8.pdf
├── README.md
├── .gitignore
├── requirements.txt
```
---
## 4. **Key Technologies**
- Streamlit – Fast web deployment for all components
- PyTorch – Model training & evaluation 
- Timm – Pretrained models like EfficientNet, Swin Transformer
- OpenCV – Image processing & CLAHE contrast enhancement
- PyTorch Grad-CAM – Model interpretability
- Pandas, NumPy, Matplotlib, Seaborn – Data wrangling and visualization
- RAG (Retrieval-Augmented Generation) – Powering the chatbot pipeline
---
## 5. **How to Run the Application**
### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/eye-vision.git
   cd eye-vision
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory and add your API keys:
   ```
   # Google Cloud API Key for translation services
   GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   
   # Other API keys if needed
   ```
## 🖥️ Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app1.py
   ```

2. **Access the application**
    Open your web browser and navigate to `http://localhost:8501`

---   

## 6. **Future Work**

- 🔬 Blood vessel segmentation
- 🤖 Meta model using combined LAB + grayscale spaces
- 👁️ Optic disc detection
- 🧠 Semi-supervised labeling of unlabeled segmentation datasets
- 🔗 Fusion of detection + segmentation in a hybrid model

---
## 7. **Let’s Collaborate**

This project is open to collaboration from AI engineers, clinicians, and researchers. Feel free to contribute models, datasets, or domain expertise to improve and validate this tool further.

---
## 8. **Collaborators**

* **Amany Alsayed**
  🔗 [LinkedIn](https://www.linkedin.com/in/amany-alsayed82) | 💻 [GitHub](https://github.com/Amany-alsayed) | ✉️ [Email](mailto:amanyalsayed82@gmail.com)

* **Aya Mohammedd**
  🔗 [LinkedIn](https://www.linkedin.com/in/aya-mohammed01) | 💻 [GitHub](https://github.com/AyaMohammedd) | ✉️ [Email](mailto:raya80224@gmail.com)

* **Sara Elwatany**
  🔗 [LinkedIn](https://www.linkedin.com/in/sara-elwatany) | 💻 [GitHub](https://github.com/SaraElwatany) | ✉️ [Email](mailto:saraayman10000@gmail.com)

* **Shaza Osama**
  🔗 [LinkedIn](https://www.linkedin.com/in/shaza-osama-196390211) | 💻 [GitHub](https://github.com/ShazaOsama785) | ✉️ [Email](mailto:shazaosama785@gmail.com)

* **Zad Walid**
  🔗 [LinkedIn](https://www.linkedin.com/in/zadwalid) | 💻 [GitHub](https://github.com/Zad-Walid) | ✉️ [Email](mailto:zadwalid06@gmail.com)

