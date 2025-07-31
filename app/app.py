import streamlit as st
from PIL import Image
import datetime
from fpdf import FPDF
from deep_translator import GoogleTranslator
import os
import time
import cv2
import io
import base64
import numpy as np
import random
from report_generator import generate_pdf
from explainability.grad_cam import inference as grad_cam_inference
from segmentation.exudates_inference import predict_exudates, load_model
from chatbot.chatbot import send_to_gimini_with_rag 
from dotenv import load_dotenv
load_dotenv()

# Set page config first for better layout
import streamlit as st

st.set_page_config(
    page_title="Eye Vision",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("app\style.css")

LANG_MAP = {
    "English": "en",
    "Arabic": "ar",
    "French": "fr",
    "Spanish": "es"
}

TRANSLATIONS = {
    "en": {
        "Language": "Language",
        "Choose language": "Choose language",
        "Get Started": "Get Started",
        "Upload Retinal Image": "Upload Retinal Image",
        "Take a picture of the retina": "Take a picture of the retina",
        "Select image file": "Select image file",
        "Image Guidelines": "Image Guidelines",
        "Back to Home": "Back to Home",
        "AI Analysis": "AI Analysis",
        "No image uploaded. Please upload an image first.": "No image uploaded. Please upload an image first.",
        "Go to Upload": "Go to Upload",
        "Uploaded Image": "Uploaded Image",
        "Retinal Image for Analysis": "Retinal Image for Analysis",
        "Analysis Controls": "Analysis Controls",
        "Start AI Analysis": "Start AI Analysis",
        "AI is analyzing the retinal image...": "AI is analyzing the retinal image...",
        "Upload Different Image": "Upload Different Image",
        "Analysis Results": "Analysis Results",
        "Analyzed Image": "Analyzed Image",
        "Retinal Image Analysis": "Retinal Image Analysis",
        "Diagnosis Results": "Diagnosis Results",
        "Immediate Actions Required": "Immediate Actions Required",
        "Recommended Follow-up": "Recommended Follow-up",
        "Download Detailed Report": "Download Detailed Report",
        "Analyze Another Image": "Analyze Another Image",
        "Chat with AI Assistant": "Chat with AI Assistant",
        "AI Medical Assistant": "AI Medical Assistant",
        "Quick Questions": "Quick Questions",
        "What is diabetic retinopathy?": "What is diabetic retinopathy?",
        "What are the symptoms?": "What are the symptoms?",
        "How is it treated?": "How is it treated?",
        "Screening frequency?": "Screening frequency?",
        "Ask Your Question": "Ask Your Question",
        "Type your question about diabetic retinopathy...": "Type your question about diabetic retinopathy...",
        "Conversation History": "Conversation History",
        "No conversation yet. Ask a question to get started!": "No conversation yet. Ask a question to get started!",
        "You": "You",
        "AI Assistant": "AI Assistant",
        "About Eye Vision": "About Eye Vision",
        "Navigation": "Navigation",
        "Home": "Home",
        "Upload": "Upload",
        "Analysis": "Analysis",
        "Results": "Results",
        "Chat": "Chat",
        "About": "About",
        "Select Page": "Select Page",
        "Diabetic Retinopathy Detected": "Diabetic Retinopathy Detected",
        "No Diabetic Retinopathy Detected": "No Diabetic Retinopathy Detected",
        "AI-Powered Diabetic Retinopathy Detection System": "AI-Powered Diabetic Retinopathy Detection System",
        "Advanced AI Detection": "Advanced AI Detection",
        "Our cutting-edge deep learning models analyze retinal images with medical-grade accuracy to detect early signs of diabetic retinopathy.": "Our cutting-edge deep learning models analyze retinal images with medical-grade accuracy to detect early signs of diabetic retinopathy.",
        "Instant Results": "Instant Results",
        "Get comprehensive analysis results within seconds, complete with detailed reports and medical recommendations.": "Get comprehensive analysis results within seconds, complete with detailed reports and medical recommendations.",
        "Multi-Language Support": "Multi-Language Support",
        "Available in multiple languages to serve healthcare providers and patients worldwide.": "Available in multiple languages to serve healthcare providers and patients worldwide.",
        "This system is designed to assist healthcare professionals in early detection of diabetic retinopathy. Always consult with qualified medical professionals for diagnosis and treatment.": "This system is designed to assist healthcare professionals in early detection of diabetic retinopathy. Always consult with qualified medical professionals for diagnosis and treatment.",
        "Provide a clear retinal image for AI analysis": "Provide a clear retinal image for AI analysis",
        "Camera Capture": "Camera Capture",
        "Take a live photo using your device camera": "Take a live photo using your device camera",
        "File Upload": "File Upload",
        "Upload an image from your device gallery": "Upload an image from your device gallery",
        "High Resolution": "High Resolution",
        "Use images with at least 1024x1024 pixels for best results": "Use images with at least 1024x1024 pixels for best results",
        "Good Lighting": "Good Lighting",
        "Ensure proper illumination without glare or shadows": "Ensure proper illumination without glare or shadows",
        "Centered Focus": "Centered Focus",
        "Center the optic disc and macula in the image": "Center the optic disc and macula in the image",
        "Advanced diabetic retinopathy detection": "Advanced diabetic retinopathy detection",
        "Image Details": "Image Details",
        "Size": "Size",
        "Format": "Format",
        "Mode": "Mode",
        "AI Analysis Process": "AI Analysis Process",
        "Image preprocessing and enhancement": "Image preprocessing and enhancement",
        "Feature extraction using deep learning": "Feature extraction using deep learning",
        "Pattern recognition and classification": "Pattern recognition and classification",
        "Confidence scoring and reporting": "Confidence scoring and reporting",
        "Comprehensive diabetic retinopathy assessment": "Comprehensive diabetic retinopathy assessment",
        "No analysis results found. Please run analysis first.": "No analysis results found. Please run analysis first.",
        "Go to Analysis": "Go to Analysis",
        "Confidence Score": "Confidence Score",
        "Risk Level": "Risk Level",
        "Recommendations": "Recommendations",
        "Consult an ophthalmologist immediately": "Consult an ophthalmologist immediately",
        "Schedule regular eye exams": "Schedule regular eye exams",
        "Monitor blood sugar levels": "Monitor blood sugar levels",
        "Follow prescribed treatment plan": "Follow prescribed treatment plan",
        "Report Generated": "Report Generated",
        "Detailed medical report with findings and recommendations": "Detailed medical report with findings and recommendations",
        "High": "High",
        "Low": "Low",
        "Send": "Send",
        "Clear Chat": "Clear Chat"
    },
    "ar": {
        "Language": "اللغة",
        "Choose language": "اختر اللغة",
        "Get Started": "ابدأ",
        "Upload Retinal Image": "رفع صورة الشبكية",
        "Take a picture of the retina": "التقط صورة للشبكية",
        "Select image file": "اختر ملف الصورة",
        "Image Guidelines": "إرشادات الصورة",
        "Back to Home": "العودة للرئيسية",
        "AI Analysis": "تحليل الذكاء الاصطناعي",
        "No image uploaded. Please upload an image first.": "لم يتم رفع صورة. يرجى رفع صورة أولاً.",
        "Go to Upload": "اذهب للرفع",
        "Uploaded Image": "الصورة المرفوعة",
        "Retinal Image for Analysis": "صورة الشبكية للتحليل",
        "Analysis Controls": "أدوات التحليل",
        "Start AI Analysis": "بدء تحليل الذكاء الاصطناعي",
        "AI is analyzing the retinal image...": "الذكاء الاصطناعي يحلل صورة الشبكية...",
        "Upload Different Image": "رفع صورة مختلفة",
        "Analysis Results": "نتائج التحليل",
        "Analyzed Image": "الصورة المحللة",
        "Retinal Image Analysis": "تحليل صورة الشبكية",
        "Diagnosis Results": "نتائج التشخيص",
        "Immediate Actions Required": "إجراءات فورية مطلوبة",
        "Recommended Follow-up": "المتابعة الموصى بها",
        "Download Detailed Report": "تحميل التقرير المفصل",
        "Analyze Another Image": "تحليل صورة أخرى",
        "Chat with AI Assistant": "المحادثة مع المساعد الذكي",
        "AI Medical Assistant": "المساعد الطبي الذكي",
        "Quick Questions": "أسئلة سريعة",
        "What is diabetic retinopathy?": "ما هو اعتلال الشبكية السكري؟",
        "What are the symptoms?": "ما هي الأعراض؟",
        "How is it treated?": "كيف يتم العلاج؟",
        "Screening frequency?": "تكرار الفحص؟",
        "Ask Your Question": "اسأل سؤالك",
        "Type your question about diabetic retinopathy...": "اكتب سؤالك عن اعتلال الشبكية السكري...",
        "Conversation History": "تاريخ المحادثة",
        "No conversation yet. Ask a question to get started!": "لا توجد محادثة بعد. اسأل سؤالاً للبدء!",
        "You": "أنت",
        "AI Assistant": "المساعد الذكي",
        "About Eye Vision": "عن Eye Vision",
        "Navigation": "التنقل",
        "Home": "الرئيسية",
        "Upload": "رفع",
        "Analysis": "التحليل",
        "Results": "النتائج",
        "Chat": "المحادثة",
        "About": "حول",
        "Select Page": "اختر الصفحة",
        "Diabetic Retinopathy Detected": "تم اكتشاف اعتلال الشبكية السكري",
        "No Diabetic Retinopathy Detected": "لم يتم اكتشاف اعتلال الشبكية السكري",
        "Send": "إرسال",
        "Clear Chat": "مسح المحادثة"
    },
    "fr": {
        "Language": "Langue",
        "Choose language": "Choisir la langue",
        "Get Started": "Commencer",
        "Upload Retinal Image": "Télécharger une image rétinienne",
        "Take a picture of the retina": "Prendre une photo de la rétine",
        "Select image file": "Sélectionner un fichier image",
        "Image Guidelines": "Directives pour l'image",
        "Back to Home": "Retour à l'accueil",
        "AI Analysis": "Analyse IA",
        "No image uploaded. Please upload an image first.": "Aucune image téléchargée. Veuillez d'abord télécharger une image.",
        "Go to Upload": "Aller au téléchargement",
        "Uploaded Image": "Image téléchargée",
        "Retinal Image for Analysis": "Image rétinienne pour analyse",
        "Analysis Controls": "Contrôles d'analyse",
        "Start AI Analysis": "Commencer l'analyse IA",
        "AI is analyzing the retinal image...": "L'IA analyse l'image rétinienne...",
        "Upload Different Image": "Télécharger une image différente",
        "Analysis Results": "Résultats d'analyse",
        "Analyzed Image": "Image analysée",
        "Retinal Image Analysis": "Analyse d'image rétinienne",
        "Diagnosis Results": "Résultats du diagnostic",
        "Immediate Actions Required": "Actions immédiates requises",
        "Recommended Follow-up": "Suivi recommandé",
        "Download Detailed Report": "Télécharger le rapport détaillé",
        "Analyze Another Image": "Analyser une autre image",
        "Chat with AI Assistant": "Discuter avec l'assistant IA",
        "AI Medical Assistant": "Assistant médical IA",
        "Quick Questions": "Questions rapides",
        "What is diabetic retinopathy?": "Qu'est-ce que la rétinopathie diabétique?",
        "What are the symptoms?": "Quels sont les symptômes?",
        "How is it treated?": "Comment est-elle traitée?",
        "Screening frequency?": "Fréquence de dépistage?",
        "Ask Your Question": "Posez votre question",
        "Type your question about diabetic retinopathy...": "Tapez votre question sur la rétinopathie diabétique...",
        "Conversation History": "Historique de conversation",
        "No conversation yet. Ask a question to get started!": "Pas encore de conversation. Posez une question pour commencer!",
        "You": "Vous",
        "AI Assistant": "Assistant IA",
        "About Eye Vision": "À propos d'Eye",
        "Navigation": "Navigation",
        "Home": "Accueil",
        "Upload": "Télécharger",
        "Analysis": "Analyse",
        "Results": "Résultats",
        "Chat": "Discussion",
        "About": "À propos",
        "Select Page": "Sélectionner la page",
        "Diabetic Retinopathy Detected": "Rétinopathie diabétique détectée",
        "No Diabetic Retinopathy Detected": "Aucune rétinopathie diabétique détectée",
        "Send": "Envoyer",
        "Clear Chat": "Effacer la discussion"
    },
    "es": {
        "Language": "Idioma",
        "Choose language": "Elegir idioma",
        "Get Started": "Empezar",
        "Upload Retinal Image": "Subir imagen retinal",
        "Take a picture of the retina": "Tomar una foto de la retina",
        "Select image file": "Seleccionar archivo de imagen",
        "Image Guidelines": "Pautas de imagen",
        "Back to Home": "Volver al inicio",
        "AI Analysis": "Análisis IA",
        "No image uploaded. Please upload an image first.": "No se ha subido ninguna imagen. Por favor, suba una imagen primero.",
        "Go to Upload": "Ir a subir",
        "Uploaded Image": "Imagen subida",
        "Retinal Image for Analysis": "Imagen retinal para análisis",
        "Analysis Controls": "Controles de análisis",
        "Start AI Analysis": "Iniciar análisis IA",
        "AI is analyzing the retinal image...": "La IA está analizando la imagen retinal...",
        "Upload Different Image": "Subir imagen diferente",
        "Analysis Results": "Resultados del análisis",
        "Analyzed Image": "Imagen analizada",
        "Retinal Image Analysis": "Análisis de imagen retinal",
        "Diagnosis Results": "Resultados del diagnóstico",
        "Immediate Actions Required": "Acciones inmediatas requeridas",
        "Recommended Follow-up": "Seguimiento recomendado",
        "Download Detailed Report": "Descargar informe detallado",
        "Analyze Another Image": "Analizar otra imagen",
        "Chat with AI Assistant": "Chatear con asistente IA",
        "AI Medical Assistant": "Asistente médico IA",
        "Quick Questions": "Preguntas rápidas",
        "What is diabetic retinopathy?": "¿Qué es la retinopatía diabética?",
        "What are the symptoms?": "¿Cuáles son los síntomas?",
        "How is it treated?": "¿Cómo se trata?",
        "Screening frequency?": "¿Frecuencia de detección?",
        "Ask Your Question": "Haga su pregunta",
        "Type your question about diabetic retinopathy...": "Escriba su pregunta sobre retinopatía diabética...",
        "Conversation History": "Historial de conversación",
        "No conversation yet. Ask a question to get started!": "¡Aún no hay conversación. Haga una pregunta para empezar!",
        "You": "Usted",
        "AI Assistant": "Asistente IA",
        "About Eye Vision": "Acerca de Eye Vision",
        "Navigation": "Navegación",
        "Home": "Inicio",
        "Upload": "Subir",
        "Analysis": "Análisis",
        "Results": "Resultados",
        "Chat": "Chat",
        "About": "Acerca de",
        "Select Page": "Seleccionar página",
        "Diabetic Retinopathy Detected": "Retinopatía diabética detectada",
        "No Diabetic Retinopathy Detected": "No se detectó retinopatía diabética",
        "Send": "Enviar",
        "Clear Chat": "Limpiar chat"
    }
}

# Helper function to get translated text
def get_text(key):
    """Get translated text based on current language"""
    current_lang = LANG_MAP.get(st.session_state.language, "en")
    return TRANSLATIONS.get(current_lang, {}).get(key, key)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Splash"
if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "language" not in st.session_state:
    st.session_state.language = "English"


    
def T(text):
    lang = st.session_state.get("language", "English")
    lang_code = LANG_MAP[lang]
    if lang_code == "en":
        return text
    if lang_code in TRANSLATIONS and text in TRANSLATIONS[lang_code]:
        return TRANSLATIONS[lang_code][text]
    return text 

def translate_ui():
    st.sidebar.markdown("### 🌍 " + T("Language"))
    current_lang = st.session_state.get("language", "English")
    lang_options = list(LANG_MAP.keys())
    current_index = lang_options.index(current_lang) if current_lang in lang_options else 0
    selected = st.sidebar.selectbox(
        T("Choose language"),
        lang_options,
        index=current_index,
        key="language_selector"
    )
    if selected != st.session_state.get("language", "English"):
        st.session_state.language = selected
        st.rerun()

def navigation_sidebar():
    st.sidebar.markdown("### 🧣 " + T("Navigation"))
    
    # Add Chat to your navigation options
    nav_options = {
        "🏠 " + T("Home"): "Splash",
        "📄 " + T("Upload"): "Workflow",
        "💬 " + T("Chat"): "Chat",  # Add this line
        "ℹ️ " + T("About"): "About"
    }
    
    selected = st.sidebar.radio(
        T("Select Page"),
        list(nav_options.keys()),
        index=list(nav_options.values()).index(st.session_state.page) if st.session_state.page in nav_options.values() else 0
    )
    
    if nav_options[selected] != st.session_state.page:
        st.session_state.page = nav_options[selected]
        st.rerun()

def splash_screen():
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">👁️ Eye Vision</div>
        <div class="hero-subtitle">{T("AI-Powered Diabetic Retinopathy Detection System")}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Info text moved up, before feature cards
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="
                background-color: #f0f4ff;
                color: #2c3e50;
                padding: 1rem;
                border-radius: 8px;
                border-right: 4px solid #667eea;
                font-size: 0.95rem;
                line-height: 1.5;
                margin: 0 0 2rem 0;
            ">
                <div style="
                    display: flex;
                    align-items: flex-start;
                    gap: 10px;
                ">
                    <span style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        -webkit-background-clip: text;
                        background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-size: 1.2em;
                    ">💡</span>
                    <span>""" + T("This system is designed to assist healthcare professionals in early detection of diabetic retinopathy. Always consult with qualified medical professionals for diagnosis and treatment.") + """
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Feature cards
        st.markdown(f"""
        <div class="feature-card">
            <h3>🌟 {T("Advanced AI Detection")}</h3>
            <p>{T("Our cutting-edge deep learning models analyze retinal images with medical-grade accuracy to detect early signs of diabetic retinopathy.")}</p>
        </div>
        <div class="feature-card">
            <h3>⚡ {T("Instant Results")}</h3>
            <p>{T("Get comprehensive analysis results within seconds, complete with detailed reports and medical recommendations.")}</p>
        </div>
        <div class="feature-card">
            <h3>🌐 {T("Multi-Language Support")}</h3>
            <p>{T("Available in multiple languages to serve healthcare providers and patients worldwide.")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 " + T("Get Started"), key="start_btn_splash"):
            st.session_state.page = "Workflow"
            st.rerun()
        st.markdown("---")

def workflow_page():
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">👁️ Eye Vision</div>
        <div class="hero-subtitle">{T("AI-Powered Diabetic Retinopathy Detection System")}</div>
    </div>
    """, unsafe_allow_html=True)

    # Single upload section
    if st.session_state.uploaded_img is None:
        st.markdown(f"### {T('Upload Retinal Image')}")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"#### 📸 {T('Camera Capture')}")
            img_file_cam = st.camera_input(T("Take a picture of the retina"), key="camera_input")
        with col2:
            st.markdown(f"#### 🖼️ {T('File Upload')}")
            img_file_up = st.file_uploader(T("Select image file"), type=["jpg", "jpeg", "png"], key="file_uploader")
        
        uploaded_img = img_file_cam or img_file_up
        if uploaded_img:
            try:
                img = Image.open(uploaded_img)
                if img.size[0] < 512 or img.size[1] < 512:
                    st.warning(T("Image is too small. Please upload an image at least 512x512 pixels."))
                else:
                    st.session_state.uploaded_img = img
                    st.rerun()
            except Exception:
                st.error(T("Invalid image file. Please upload a valid image."))
        
        # Show guidelines with enhanced styling
        st.markdown("### 📋 " + T("Image Guidelines"))
        
        # Guidelines container
        st.markdown("""
        <div class="guidelines-section">
            <div class="guideline-card">
                <div class="guideline-title">
                    <span class="guideline-icon">🔍</span>
                    <span>{}</span>
                </div>
                <div class="guideline-description">
                    {}
                </div>
            </div>
            <div class="guideline-card">
                <div class="guideline-title">
                    <span class="guideline-icon">💡</span>
                    <span>{}</span>
                </div>
                <div class="guideline-description">
                    {}
                </div>
            </div>
            <div class="guideline-card">
                <div class="guideline-title">
                    <span class="guideline-icon">🎯</span>
                    <span>{}</span>
                </div>
                <div class="guideline-description">
                    {}
                </div>
            </div>
        </div>
        """.format(
            T("High Resolution"),
            T("Use images with at least 1024x1024 pixels for best results"),
            T("Good Lighting"),
            T("Ensure proper illumination without glare or shadows for optimal analysis"),
            T("Centered Focus"),
            T("Center the optic disc and macula in the image for accurate diagnosis")
        ), unsafe_allow_html=True)
        return

    # If image is uploaded, show analysis options
    if st.session_state.uploaded_img is not None:
        # Show uploaded image
        st.markdown("### 🖼️ " + T("Uploaded Image"))
        st.image(st.session_state.uploaded_img, caption=T("Retinal Image for Analysis"), use_container_width=True)
        
        # Initialize session state for analysis results if not exists
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
            
        # Model selection
        st.markdown("### 🔬 Model Selection")
        model_option = st.selectbox("Choose model", 
                                  ["MobileNet (DR Detection)", "U-Net (Exudates Segmentation)", "Combined Report"], 
                                  key="model_selector")

        # Save uploaded image temporarily
        temp_image_path = "temp_uploaded_image.jpg"
        st.session_state.uploaded_img.save(temp_image_path)
        
        # Store current model in session state
        st.session_state.current_model = model_option
        
        # Check if we have results for the current model
        current_results = st.session_state.analysis_results.get(model_option, {})
        
        # Model-specific analysis
        if model_option == "MobileNet (DR Detection)":
            run_analysis = st.button("🔍 Run DR Detection", key="run_dr_detection_btn")
            
            # Check if we need to run analysis or show existing results
            if run_analysis or current_results.get('analysis_done', False):
                try:
                    # Only run analysis if we don't have results yet
                    if run_analysis or not current_results.get('analysis_done', False):
                        with st.spinner("Analyzing image for diabetic retinopathy..."):
                            original, gradcam, prediction = grad_cam_inference(temp_image_path, model_path="models/mobilenet_dr(70%).pth")
                            
                            # Process results
                            gradcam = (gradcam * 255).astype('uint8') if gradcam.max() <= 1 else gradcam
                            original = (original * 255).astype('uint8') if original.max() <= 1 else original
                            
                            # Save results to session state
                            original_path = "original_temp.jpg"
                            gradcam_path = "gradcam_temp.jpg"
                            cv2.imwrite(original_path, cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam, cv2.COLOR_RGB2BGR))
                            
                            st.session_state.analysis_results[model_option] = {
                                'analysis_done': True,
                                'original': original,
                                'gradcam': gradcam,
                                'prediction': prediction,
                                'original_path': original_path,
                                'gradcam_path': gradcam_path
                            }
                            st.rerun()
                    else:
                        # Use cached results
                        result = st.session_state.analysis_results[model_option]
                        original = result['original']
                        gradcam = result['gradcam']
                        prediction = result['prediction']
                        original_path = result['original_path']
                        gradcam_path = result['gradcam_path']
                    
                    # Display results
                    st.subheader(f"🔍 Diagnosis: {prediction}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(gradcam, caption=f"Grad-CAM: {prediction}", use_container_width=True)

                    # Download options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        with open(original_path, "rb") as f:
                            st.download_button("📥 Download Original", 
                                            data=f.read(), 
                                            file_name="original.jpg", 
                                            mime="image/jpeg",
                                            key="dl_original")
                    with col2:
                        with open(gradcam_path, "rb") as f:
                            st.download_button("📥 Download Grad-CAM", 
                                            data=f.read(), 
                                            file_name="gradcam.jpg", 
                                            mime="image/jpeg",
                                            key="dl_gradcam")
                    
                    # PDF Report Generation
                    with col3:
                        if st.button("📄 Generate Report", key="gen_report_dr"):
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            pdf_path = f"DR_Report_{timestamp}.pdf"
                            risk_level = "High" if "Severe" in prediction or "Proliferative" in prediction else "Medium" if "Moderate" in prediction else "Low"
                            recommendations = generate_recommendations("No DR" not in prediction, risk_level)
                            
                            try:
                                # Generate PDF and create download button
                                if generate_pdf_report_mobilenet(
                                    original_path=original_path, 
                                    gradcam_path=gradcam_path, 
                                    diagnosis=prediction,
                                    confidence=0.85,  # Confidence score placeholder
                                    risk_level=risk_level,
                                    recommendations=recommendations,
                                    pdf_path=pdf_path
                                ):
                                    with open(pdf_path, "rb") as f:
                                        st.download_button("📥 Download Report", 
                                                        data=f, 
                                                        file_name=os.path.basename(pdf_path), 
                                                        mime="application/pdf",
                                                        key=f"dl_report_{timestamp}")
                                    safe_file_cleanup(pdf_path)
                                else:
                                    st.error("Failed to generate PDF report")
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                                st.error(f"Original path exists: {os.path.exists(original_path) if 'original_path' in locals() else 'N/A'}")
                                st.error(f"Grad-CAM path exists: {os.path.exists(gradcam_path) if 'gradcam_path' in locals() else 'N/A'}")
                
                except Exception as e:
                    st.error(f"Error during DR detection: {str(e)}")
                    if 'analysis_results' in st.session_state and model_option in st.session_state.analysis_results:
                        del st.session_state.analysis_results[model_option]

        elif model_option == "U-Net (Exudates Segmentation)":
            run_analysis = st.button("🔍 Run Exudates Segmentation", key="run_exudates_segmentation_btn")
            
            # Check if we need to run analysis or show existing results
            if run_analysis or current_results.get('analysis_done', False):
                try:
                    # Only run analysis if we don't have results yet
                    if run_analysis or not current_results.get('analysis_done', False):
                        with st.spinner("Analyzing image for exudates..."):
                            # Load model
                            model = load_model("models/rgb_model.pth", in_channels=3)
                            
                            # Run prediction
                            original, binary_mask, area = predict_exudates(st.session_state.uploaded_img, model=model)
                            
                            # Save results to session state
                            original_path = "original_exudates.jpg"
                            mask_path = "mask_exudates.jpg"
                            
                            # Convert to proper format for saving
                            original_img = Image.fromarray((original * 255).astype(np.uint8))
                            mask_img = Image.fromarray(binary_mask)
                            
                            original_img.save(original_path)
                            mask_img.save(mask_path)
                            
                            st.session_state.analysis_results[model_option] = {
                                'analysis_done': True,
                                'original': original,
                                'binary_mask': binary_mask,
                                'area': area,
                                'original_path': original_path,
                                'mask_path': mask_path
                            }
                            st.rerun()
                    else:
                        # Use cached results
                        result = st.session_state.analysis_results[model_option]
                        original = result['original']
                        binary_mask = result['binary_mask']
                        area = result['area']
                        original_path = result['original_path']
                        mask_path = result['mask_path']
                    
                    # Display results
                    st.subheader(f"🔍 Exudates Analysis Results")
                    st.metric("Exudate Area", f"{area} pixels")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(binary_mask, caption="Exudate Mask", use_container_width=True, clamp=True)

                    # Download options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        with open(original_path, "rb") as f:
                            st.download_button("📥 Download Original", 
                                            data=f.read(), 
                                            file_name="original_exudates.jpg", 
                                            mime="image/jpeg",
                                            key="dl_original_exudates")
                    with col2:
                        with open(mask_path, "rb") as f:
                            st.download_button("📥 Download Mask", 
                                            data=f.read(), 
                                            file_name="exudate_mask.jpg", 
                                            mime="image/jpeg",
                                            key="dl_mask_exudates")
                    
                    # PDF Report Generation
                    with col3:
                        if st.button("📄 Generate Report", key="gen_report_exudates"):
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            pdf_path = f"Exudates_Report_{timestamp}.pdf"
                            
                            try:
                                # Ensure we have all required paths
                                if not all(os.path.exists(p) for p in [original_path, mask_path]):
                                    raise FileNotFoundError("Required image files not found for report generation")
                                
                                # Create PDF
                                pdf = FPDF()
                                pdf.set_auto_page_break(auto=True, margin=15)
                                
                                # Add title
                                pdf.add_page()
                                pdf.set_font("Arial", 'B', 16)
                                pdf.cell(0, 10, "Exudates Analysis Report", 0, 1, 'C')
                                pdf.ln(10)
                            
                                # Add analysis details
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(0, 10, "Analysis Details:", 0, 1)
                                pdf.set_font("Arial", '', 10)
                                pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
                                pdf.cell(0, 10, f"Exudate Area: {area} pixels", 0, 1)
                                pdf.ln(10)
                                
                                # Add images side by side
                                pdf.ln(10)
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(95, 10, "Original Image", 0, 0, 'C')
                                pdf.cell(95, 10, "Exudate Mask", 0, 1, 'C')
                                
                                # Add images
                                pdf.image(original_path, x=10, y=pdf.get_y(), w=90, h=0)
                                pdf.image(mask_path, x=110, y=pdf.get_y(), w=90, h=0)
                                
                                # Add recommendations
                                pdf.add_page()
                                pdf.set_font("Arial", 'B', 14)
                                pdf.cell(0, 10, "Recommendations:", 0, 1)
                                pdf.ln(5)
                                
                                risk_level = "High" if area > 5000 else "Medium" if area > 1000 else "Low"
                                recommendations = generate_recommendations(True, risk_level)
                                
                                pdf.set_font("Arial", '', 12)
                                for i, rec in enumerate(recommendations, 1):
                                    pdf.multi_cell(0, 10, f"{i}. {rec}")
                                
                                # Save PDF
                                pdf.output(pdf_path)
                                
                                # Offer download
                                with open(pdf_path, "rb") as f:
                                    st.download_button("📥 Download Report", 
                                                    data=f, 
                                                    file_name=pdf_path, 
                                                    mime="application/pdf",
                                                    key=f"dl_report_exudates_{timestamp}")
                                
                                # Clean up
                                safe_file_cleanup(pdf_path)
                
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                                return  # Exit the function if there was an error
                
                except Exception as e:
                    st.error(f"Error during exudates segmentation: {str(e)}")
                    if 'analysis_results' in st.session_state and model_option in st.session_state.analysis_results:
                        del st.session_state.analysis_results[model_option]
                                
                except Exception as e:
                    st.error(f"Error during exudates segmentation: {str(e)}")

        elif model_option == "Combined Report":
            run_analysis = st.button("🔍 Run Combined Analysis", key="run_combined_analysis_btn")
            
            # Check if we need to run analysis or show existing results
            if run_analysis or current_results.get('analysis_done', False):
                try:
                    # Only run analysis if we don't have results yet
                    if run_analysis or not current_results.get('analysis_done', False):
                        with st.spinner("Running combined analysis..."):
                            # DR Detection
                            original_dr, gradcam, prediction = grad_cam_inference(temp_image_path, model_path="models/mobilenet_dr(70%).pth")
                            gradcam = (gradcam * 255).astype('uint8') if gradcam.max() <= 1 else gradcam
                            original_dr = (original_dr * 255).astype('uint8') if original_dr.max() <= 1 else original_dr

                            # Exudates Segmentation
                            model = load_model("models/rgb_model.pth", in_channels=3)
                            original_ex, binary_mask, area = predict_exudates(st.session_state.uploaded_img, model=model)
                            
                            # Save results to session state
                            original_dr_path = "original_dr.jpg"
                            gradcam_path = "gradcam_dr.jpg"
                            original_ex_path = "original_ex.jpg"
                            mask_path = "mask_ex.jpg"
                            
                            cv2.imwrite(original_dr_path, cv2.cvtColor(original_dr, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam, cv2.COLOR_RGB2BGR))
                            
                            original_ex_img = Image.fromarray((original_ex * 255).astype(np.uint8))
                            mask_img = Image.fromarray(binary_mask)
                            original_ex_img.save(original_ex_path)
                            mask_img.save(mask_path)
                            
                            st.session_state.analysis_results[model_option] = {
                                'analysis_done': True,
                                'original_dr': original_dr,
                                'gradcam': gradcam,
                                'prediction': prediction,
                                'original_ex': original_ex,
                                'binary_mask': binary_mask,
                                'area': area,
                                'original_dr_path': original_dr_path,
                                'gradcam_path': gradcam_path,
                                'original_ex_path': original_ex_path,
                                'mask_path': mask_path
                            }
                            st.rerun()
                    else:
                        # Use cached results
                        result = st.session_state.analysis_results[model_option]
                        original_dr = result['original_dr']
                        gradcam = result['gradcam']
                        prediction = result['prediction']
                        original_ex = result['original_ex']
                        binary_mask = result['binary_mask']
                        area = result['area']
                        original_dr_path = result['original_dr_path']
                        gradcam_path = result['gradcam_path']
                        original_ex_path = result['original_ex_path']
                        mask_path = result['mask_path']
                    
                    # Display results
                    st.subheader("🔍 Combined Analysis Results")
                    
                    # DR Results
                    st.markdown("#### Diabetic Retinopathy Detection")
                    st.write(f"**Diagnosis:** {prediction}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_dr, caption="Original Image",use_container_width=True)
                    with col2:
                        st.image(gradcam, caption=f"Grad-CAM: {prediction}", use_container_width=True)
                    
                    # Exudates Results
                    st.markdown("#### Exudates Segmentation")
                    st.metric("Exudate Area", f"{area} pixels")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_ex, caption="Original Image", use_column_width=True)
                    with col2:
                        st.image(binary_mask, caption="Exudate Mask", use_column_width=True, clamp=True)
                    
                    # Combined report generation
                    if st.button("📄 Generate Combined Report", key="gen_report_combined"):
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        pdf_path = f"Combined_Report_{timestamp}.pdf"
                        
                        # Generate combined PDF
                        pdf = FPDF()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        
                        # Add DR section
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 16)
                        pdf.cell(0, 10, "Diabetic Retinopathy Analysis", 0, 1, 'C')
                        pdf.ln(10)
                        
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(0, 10, f"Diagnosis: {prediction}", 0, 1)
                        pdf.ln(5)
                        
                        # Add DR images side by side
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(95, 10, "Original Image", 0, 0, 'C')
                        pdf.cell(95, 10, "Grad-CAM Heatmap", 0, 1, 'C')
                        
                        pdf.image(original_dr_path, x=10, y=pdf.get_y(), w=90, h=0)
                        pdf.image(gradcam_path, x=110, y=pdf.get_y(), w=90, h=0)
                        
                        # Add Exudates section
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 16)
                        pdf.cell(0, 10, "Exudates Analysis", 0, 1, 'C')
                        pdf.ln(10)
                        
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(0, 10, f"Exudate Area: {area} pixels", 0, 1)
                        pdf.ln(5)
                        
                        # Add Exudates images side by side
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(95, 10, "Original Image", 0, 0, 'C')
                        pdf.cell(95, 10, "Exudate Mask", 0, 1, 'C')
                        
                        pdf.image(original_ex_path, x=10, y=pdf.get_y(), w=90, h=0)
                        pdf.image(mask_path, x=110, y=pdf.get_y(), w=90, h=0)
                        
                        # Add recommendations
                        risk_level = "High" if area > 5000 or "Severe" in prediction or "Proliferative" in prediction else "Medium" if area > 1000 or "Moderate" in prediction else "Low"
                        recommendations = generate_recommendations(True, risk_level)
                        
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 16)
                        pdf.cell(0, 10, "Recommendations", 0, 1, 'C')
                        pdf.ln(10)
                        
                        pdf.set_font("Arial", '', 12)
                        for i, rec in enumerate(recommendations, 1):
                            pdf.multi_cell(0, 10, f"{i}. {rec}")
                        
                        # Add footer
                        pdf.set_y(-15)
                        pdf.set_font("Arial", 'I', 8)
                        pdf.cell(0, 10, "This report was generated by Eye Vision - AI-Powered Diabetic Retinopathy Detection System", 0, 0, 'C')
                        
                        pdf.output(pdf_path)
                        
                        # Offer download
                        with open(pdf_path, "rb") as f:
                            st.download_button("📥 Download Combined Report", 
                                            data=f, 
                                            file_name=pdf_path, 
                                            mime="application/pdf",
                                            key=f"dl_combined_report_{timestamp}")
                        
                        # Clean up temporary files
                        safe_file_cleanup(pdf_path)
                
                except Exception as e:
                    st.error(f"Error during combined analysis: {str(e)}")
                    if 'analysis_results' in st.session_state and model_option in st.session_state.analysis_results:
                        del st.session_state.analysis_results[model_option]

        # Option to upload different image
        if st.button("🔄 " + T("Upload Different Image"), key="upload_different_btn"):
            st.session_state.uploaded_img = None
            # Clean up temp file
            safe_file_cleanup(temp_image_path)
            st.rerun()
            st.rerun()

def safe_file_cleanup(file_path):
    """Safely remove temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

def generate_pdf_report_unet(original_path, mask_path, exudates_count, severity, recommendations, pdf_path="exudates_report.pdf"):
    """
    Generate a PDF report for U-Net exudates segmentation results.
    
    Args:
        original_path: Path to the original image
        mask_path: Path to the segmentation mask image
        exudates_count: Number of exudates detected
        severity: Severity level of exudates
        recommendations: List of recommendation strings
        pdf_path: Output path for the PDF file
        
    Returns:
        Path to the generated PDF file
    """
    try:
        from fpdf import FPDF
        
        # Get current date and time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create PDF with smaller margins for more compact layout
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.add_page()
        
        # Set document metadata
        pdf.set_creator("Eye Vision - AI-Powered Diabetic Retinopathy Detection")
        pdf.set_title("Exudates Segmentation Report")
        pdf.set_author("Eye Vision System")
        
        # Add title with date/time
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 10, "Exudates Segmentation Report", 0, 1, 'C')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 5, f"Generated on: {current_time}", 0, 1, 'C')
        pdf.ln(10)
        
        # Add analysis details section
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Analysis Details", 0, 1, 'L')
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 7, f"- Exudates Detected: {exudates_count}", 0, 1, 'L')
        pdf.cell(0, 7, f"- Severity: {severity}", 0, 1, 'L')
        
        # Add a small separator
        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)
        pdf.cell(0, 1, "", 'T')
        pdf.ln(10)
        
        # Add images section - stacked vertically
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Segmentation Results", 0, 1, 'L')
        
        # Original Image
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Original Image", 0, 1, 'L')
        pdf.image(original_path, w=180)
        pdf.ln(10)
        
        # Segmentation Mask
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Exudates Segmentation", 0, 1, 'L')
        pdf.image(mask_path, w=180)
        pdf.ln(10)
        
        # Recommendations section
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Recommendations:", 0, 1, 'L')
        pdf.ln(2)
        
        pdf.set_font("Arial", '', 12)
        for rec in recommendations:
            # Use dash instead of bullet point for better compatibility
            pdf.cell(5, 7, "", 0, 0, 'L')
            pdf.multi_cell(0, 7, f"- {rec}")
            pdf.ln(2)  # Small space between recommendations
        
        # Add footer with page number
        pdf.set_y(-15)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, 'C')
        
        # Add footer text on the right
        pdf.set_x(-100)
        pdf.cell(0, 10, "Generated by Eye Vision System", 0, 0, 'R')
        
        # Save PDF
        pdf.output(pdf_path)
        return pdf_path
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None

def generate_pdf_report_mobilenet(original_path, gradcam_path, diagnosis, confidence, risk_level, recommendations, pdf_path="diagnosis_report.pdf"):
    """
    Generate a PDF report for MobileNet DR detection results.
    
    Args:
        original_path: Path to the original image
        gradcam_path: Path to the Grad-CAM heatmap image
        diagnosis: String with the diagnosis result
        confidence: Confidence score (0-1)
        risk_level: String indicating risk level (Low/Medium/High)
        recommendations: List of recommendation strings
        pdf_path: Output path for the PDF file
        
    Returns:
        Path to the generated PDF file
    """
    try:
        from fpdf import FPDF
        
        # Get current date and time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create PDF with smaller margins for more compact layout
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.add_page()
        
        # Set document metadata and enable UTF-8 support
        pdf.set_creator("Eye Vision - AI-Powered Diabetic Retinopathy Detection")
        pdf.set_title("Diabetic Retinopathy Analysis Report")
        pdf.set_author("Eye Vision System")
        
        # Add title with date/time
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 10, "Diabetic Retinopathy Analysis Report", 0, 1, 'C')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 5, f"Generated on: {current_time}", 0, 1, 'C')
        pdf.ln(10)
        
        # Add analysis details section
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Analysis Details", 0, 1, 'L')
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 7, f"- Diagnosis: {diagnosis}", 0, 1, 'L')
        pdf.cell(0, 7, f"- Confidence: {confidence*100:.1f}%", 0, 1, 'L')
        pdf.cell(0, 7, f"- Risk Level: {risk_level}", 0, 1, 'L')
        
        # Add a small separator
        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)
        pdf.cell(0, 1, "", 'T')
        pdf.ln(10)
        
        # Add images section - stacked vertically
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Analysis Results", 0, 1, 'L')
        
        # Original Image
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Original Image", 0, 1, 'L')
        pdf.image(original_path, w=180)
        pdf.ln(10)
        
        # Grad-CAM Heatmap
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Grad-CAM Heatmap", 0, 1, 'L')
        pdf.image(gradcam_path, w=180)
        pdf.ln(10)
        
        # Removed Exudates Segmentation section as it's not applicable for MobileNet model
        # and was causing 'mask_path not defined' error
        
        # Recommendations section
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Recommendations:", 0, 1, 'L')
        pdf.ln(2)
        
        pdf.set_font("Arial", '', 12)
        for rec in recommendations:
            # Use dash instead of bullet point for better compatibility
            pdf.cell(5, 7, "", 0, 0, 'L')
            pdf.multi_cell(0, 7, f"- {rec}")
            pdf.ln(2)  # Small space between recommendations
        
        # Add footer with page number
        pdf.set_y(-15)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, 'C')
        
        # Add footer text on the right
        pdf.set_x(-100)
        pdf.cell(0, 10, "Generated by Eye Vision System", 0, 0, 'R')
        
        # Save PDF
        pdf.output(pdf_path)
        return pdf_path
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None



    # 2. Preview and Analysis
    if st.session_state.prediction is None:
        st.markdown("### 🖼️ " + T("Preview"))
        st.image(st.session_state.uploaded_img, caption=T("Retinal Image Preview"), use_column_width=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("❌ " + T("Remove Image"), key="remove_img_btn"):
                st.session_state.uploaded_img = None
                st.rerun()
        with col2:
            if st.button("🔍 " + T("Start AI Analysis"), key="start_analysis_btn"):
                with st.spinner(T("AI is analyzing the retinal image...")):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    prediction_result = simulate_ai_prediction()
                    st.session_state.prediction = prediction_result
                    st.rerun()
        return

    # 3. Results
    prediction = st.session_state.prediction
    st.markdown(f"### 📊 {T('Analysis Results')}")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(st.session_state.uploaded_img, caption=T("Retinal Image Analysis"), use_column_width=True)
    with col2:
        if prediction["has_retinopathy"]:
            st.markdown(f"<div class='result-warning'><h3>⚠️ {T('Diabetic Retinopathy Detected')}</h3></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-success'><h3>✅ {T('No Diabetic Retinopathy Detected')}</h3></div>", unsafe_allow_html=True)
        st.markdown(f"**{T('Confidence Score')}:**")
        st.progress(prediction["confidence"])
        risk_color = "#e74c3c" if prediction["risk_level"] == "High" else "#f1c40f" if prediction["risk_level"] == "Medium" else "#2ecc71"
        st.markdown(
            f'<span style="background-color:{risk_color};color:#fff;padding:4px 12px;border-radius:12px;font-weight:bold;">'
            f'{T("Risk Level")}: {T(prediction["risk_level"])}'
            '</span>',
            unsafe_allow_html=True
        )
        if prediction.get("severity") and prediction["severity"] != "None":
            st.markdown(
                f'<span style="background-color:#8e44ad;color:#fff;padding:4px 12px;border-radius:12px;font-weight:bold;">'
                f'{T("Severity")}: {prediction["severity"]}'
                '</span>',
                unsafe_allow_html=True
            )
    st.markdown("---")
    st.markdown(f"### 📋 {T('Diagnosis Results')}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"#### 🚨 {T('Immediate Actions Required')}")
        for rec in prediction["recommendations"][:2]:
            st.markdown(f"- {T(rec)}")
    with col2:
        st.markdown(f"#### 📅 {T('Recommended Follow-up')}")
        for rec in prediction["recommendations"][2:]:
            st.markdown(f"- {T(rec)}")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 " + T("Download Detailed Report"), key="download_report_result"):
            generate_pdf_report(prediction)
    with col2:
        if st.button("🔄 " + T("Analyze Another Image"), key="analyze_another_result"):
            st.session_state.uploaded_img = None
            st.session_state.prediction = None
            st.rerun()
    with col3:
        if st.button("💬 " + T("Chat with AI Assistant"), key="go_to_chat_result"):
            st.session_state.page = "Chat"
            st.rerun()

def simulate_ai_prediction():
    has_retinopathy = random.choice([True, False])
    confidence = random.uniform(0.7, 0.95)
    if has_retinopathy:
        risk_level = random.choice(["High", "Medium"])
        severity = random.choice(["Mild", "Moderate", "Severe"])
    else:
        risk_level = "Low"
        severity = "None"
    return {
        "has_retinopathy": has_retinopathy,
        "confidence": confidence,
        "risk_level": risk_level,
        "severity": severity,
        "timestamp": datetime.datetime.now(),
        "recommendations": generate_recommendations(has_retinopathy, risk_level)
    }

def generate_recommendations(has_retinopathy, risk_level):
    if has_retinopathy:
        if risk_level == "High":
            return [
                "Consult an ophthalmologist immediately",
                "Consider laser treatment evaluation",
                "Monitor blood sugar levels closely",
                "Follow prescribed treatment plan"
            ]
        else:
            return [
                "Schedule regular eye exams",
                "Monitor blood sugar levels",
                "Follow prescribed treatment plan",
                "Maintain healthy lifestyle"
            ]
    else:
        return [
            "Continue regular eye screenings",
            "Monitor blood sugar levels",
            "Maintain healthy lifestyle",
            "Follow up in 12 months"
        ]

def chat_page():
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">💬 {T("Chat with AI Assistant")}</div>
        <div class="hero-subtitle">{T("AI Medical Assistant")}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"### ❓ {T('Quick Questions')}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🤔 " + T("What is diabetic retinopathy?"), key="q1_chat"):
            add_to_chat("user", T("What is diabetic retinopathy?"))
            with st.spinner("Getting response..."):
                add_to_chat("assistant", get_ai_response(T("What is diabetic retinopathy?")))
            st.rerun()
            
        if st.button("🔍 " + T("What are the symptoms?"), key="q2_chat"):
            add_to_chat("user", T("What are the symptoms?"))
            with st.spinner("Getting response..."):
                add_to_chat("assistant", get_ai_response(T("What are the symptoms?")))
            st.rerun()
            
    with col2:
        if st.button("💊 " + T("How is it treated?"), key="q3_chat"):
            add_to_chat("user", T("How is it treated?"))
            with st.spinner("Getting response..."):
                add_to_chat("assistant", get_ai_response(T("How is it treated?")))
            st.rerun()
            
        if st.button("⏰ " + T("Screening frequency?"), key="q4_chat"):
            add_to_chat("user", T("Screening frequency?"))
            with st.spinner("Getting response..."):
                add_to_chat("assistant", get_ai_response(T("Screening frequency?")))
            st.rerun()
    
    st.markdown(f"### 💭 {T('Ask Your Question')}")
    
    # Create a form for better input handling
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])  # Wider input, narrower button
        with col1:
            user_input = st.text_input(
                "",
                placeholder=T("Type your question about diabetic retinopathy..."),
                key="chat_input_form"
            )
        with col2:
            st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)  # Adjust vertical alignment
            send_button = st.form_submit_button("📤 " + T("Send"), use_container_width=True)
        
        if send_button and user_input.strip():
            add_to_chat("user", user_input)
            with st.spinner("Getting response..."):
                add_to_chat("assistant", get_ai_response(user_input))
            st.rerun()
    
    if st.button("🗑️ " + T("Clear Chat"), key="clear_chat_btn"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown(f"### 💬 {T('Conversation History')}")
    if not st.session_state.chat_history:
        st.info(T("No conversation yet. Ask a question to get started!"))
    else:
        # Show messages in chronological order (latest at bottom)
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message">
                    <strong>👤 {T("You")}:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message">
                    <strong>🤖 {T("AI Assistant")}:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)

def add_to_chat(role, content):
    """Add message to chat history"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": role, "content": content})

def get_ai_response(question):
    """Get response from AI chatbot"""
    try:
        print(f"Getting AI response for: {question}")  # Debug print
        response = send_to_gimini_with_rag(question)
        print(f"AI response received: {response}")  # Debug print
        return response
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return f"I'm sorry, I encountered an error: {str(e)}. Please try again later."
    
def about_page():
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">ℹ️ {T("About Eye Vision")}</div>
        <div class="hero-subtitle">Advanced AI-Powered Healthcare Solution</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="feature-card">
        <h3>🎯 Our Mission</h3>
        <p>Eye Vision is dedicated to democratizing access to early diabetic retinopathy detection through cutting-edge artificial intelligence technology. We aim to assist healthcare professionals worldwide in providing better patient care.</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <h3>🔬 Technology</h3>
            <ul>
                <li>Deep Learning Neural Networks</li>
                <li>Computer Vision Processing</li>
                <li>Medical Image Analysis</li>
                <li>Real-time AI Inference</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <h3>🌟 Features</h3>
            <ul>
                <li>High Accuracy Detection</li>
                <li>Instant Results</li>
                <li>Multi-language Support</li>
                <li>Detailed Medical Reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="feature-card">
        <h3>⚖️ Disclaimer</h3>
        <p>This system is designed as a diagnostic aid for healthcare professionals. It should not replace professional medical judgment or consultation with qualified healthcare providers. Always consult with medical professionals for proper diagnosis and treatment.</p>
    </div>
    """, unsafe_allow_html=True)

def generate_pdf_report(prediction):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Eye Vision - Diabetic Retinopathy Analysis Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Analysis Details:", ln=True)
        pdf.set_font("Arial", size=10)
        
        # Add timestamp if it exists in prediction
        if 'timestamp' in prediction:
            pdf.cell(0, 10, f"Date: {prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        
        # Add diagnosis if it exists
        if 'has_retinopathy' in prediction:
            pdf.cell(0, 10, f"Diagnosis: {'Diabetic Retinopathy Detected' if prediction['has_retinopathy'] else 'No Diabetic Retinopathy Detected'}", ln=True)
        
        # Add confidence if it exists
        if 'confidence' in prediction:
            pdf.cell(0, 10, f"Confidence Score: {prediction['confidence']:.1%}", ln=True)
        
        # Add risk level if it exists
        if 'risk_level' in prediction:
            pdf.cell(0, 10, f"Risk Level: {prediction['risk_level']}", ln=True)
        
        pdf.ln(10)
        
        # Add recommendations if they exist
        if 'recommendations' in prediction and prediction['recommendations']:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Recommendations:", ln=True)
            pdf.set_font("Arial", size=10)
            for i, rec in enumerate(prediction['recommendations'], 1):
                pdf.cell(0, 10, f"{i}. {rec}", ln=True)
        
        # Add images if they exist
        if 'original_path' in prediction and 'mask_path' in prediction and os.path.exists(prediction['mask_path']):
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Analysis Images", ln=True)
            pdf.ln(5)
            
            # Original image
            if os.path.exists(prediction['original_path']):
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Original Image", 0, 1, 'L')
                try:
                    pdf.image(prediction['original_path'], w=180)
                    pdf.ln(5)
                except Exception as img_error:
                    pdf.cell(0, 10, f"Could not load original image: {str(img_error)}", ln=True)
            
            # Mask image
            if os.path.exists(prediction['mask_path']):
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Segmentation Mask", 0, 1, 'L')
                try:
                    pdf.image(prediction['mask_path'], w=180)
                    pdf.ln(5)
                except Exception as img_error:
                    pdf.cell(0, 10, f"Could not load mask image: {str(img_error)}", ln=True)
        
        # Generate PDF
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="diabetic_retinopathy_report.pdf">📄 {T("Download Detailed Report")}</a>'
        
        st.markdown(f"""
        <div class="result-success">
            <h4>📄 {T("Report Generated")}</h4>
            <p>{T("Detailed medical report with findings and recommendations")}</p>
            {href}
        </div>
        """, unsafe_allow_html=True)
        
        return True
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return False

def main():
    translate_ui()
    navigation_sidebar()
    

    # This part handles the page navigation
    if st.session_state.page == "Splash":
        splash_screen()
    elif st.session_state.page == "Workflow":
        workflow_page()
    elif st.session_state.page == "Chat":
        chat_page()
    elif st.session_state.page == "About":
        about_page()

def cleanup_temp_files():
    """Clean up temporary files on app restart"""
    temp_files = [
        "temp_uploaded_image.jpg",
        "original_temp.jpg", 
        "gradcam_temp.jpg",
        "original_exudates.jpg",
        "mask_exudates.jpg"
    ]
    for file_path in temp_files:
        safe_file_cleanup(file_path)

if __name__ == "__main__":
    main() 