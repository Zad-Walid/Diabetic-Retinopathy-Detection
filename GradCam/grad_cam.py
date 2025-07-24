from torchvision import transforms,models
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class_map = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def apply_clahe(image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return  merged


def preprocessing(image_path, image_size=(224, 224)):
    image = cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = apply_clahe(image)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)

    original_image = image.astype(np.float32) / 255.0

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0)
    return original_image, input_tensor


def apply_gradcam(model,image_path,target_layer,device,image_size=(224,224),show=True):
    original_image,input_tensor=preprocessing(image_path,image_size)
    input_tensor=input_tensor.to(device)
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        class_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][class_idx].item()
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets =[ClassifierOutputTarget(class_idx)] 
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    gradcam_image = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    predicted_class_name = class_map.get(class_idx, "Unknown")
    if show:
        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(original_image)
        axs[0].axis('off')
        axs[0].set_title("Original Image")
        axs[1].imshow(gradcam_image)
        axs[1].axis('off')
        axs[1].set_title(f"Grad-CAM - Pred: {predicted_class_name} ({confidence:.2f})")
        plt.tight_layout()
        plt.show()

    return original_image,gradcam_image,predicted_class_name


def inference(image_path,model_path='GradCam\mobilenet_dr.pth'):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    num_classes=5
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1]=nn.Linear(model.last_channel,num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    target_layer = model.features[-1]    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    original_image,gradcam_image,predicted_class_name=apply_gradcam(
        model=model,
        image_path=image_path,
        target_layer=target_layer,
        device=device,
        image_size=(224, 224),
        show=True
    )
    return original_image,gradcam_image,predicted_class_name


original_image,gradcam_image,predicted_class_name=inference('GradCam\sample.jpg')
print(predicted_class_name)
