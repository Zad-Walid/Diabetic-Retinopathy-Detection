import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import segmentation_models_pytorch as smp




threshold = 0.5  # global threshold




def load_image(img_path, mode='rgb'):

    '''
    Load and preprocess an image in one of the following modes:
    - 'rgb': standard 3-channel RGB image
    - 'green': single green channel extracted and converted to 1-channel grayscale
    - 'lab': L channel is CLAHE enhanced, a and b channels are normalized to [-1, 1]
    
    Returns:
        image_tensor: torch.Tensor of shape [C, 512, 512]
    '''

    # Basic transform for all modes
    resize = T.Resize((512, 512))

    if mode == 'rgb':

        image = Image.open(img_path).convert("RGB")
        image = resize(image)
        image_tensor = T.ToTensor()(image)  # shape: [3, 512, 512]



    elif mode == 'gray':

        image = cv2.imread(img_path)
        green_channel = image[:, :, 1]  # extract green

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green_clahe = clahe.apply(green_channel)

        green_resized = cv2.resize(green_clahe, (512, 512))
        image_tensor = torch.tensor(green_resized / 255.0, dtype=torch.float32).unsqueeze(0)  # [1, 512, 512]



    elif mode == 'lab':

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

        L, A, B = cv2.split(image_lab)

        # Enhance contrast of L channel using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L_enhanced = clahe.apply(L)

        # Normalize A and B to [-1, 1]
        A = (A.astype(np.float32) - 128) / 128
        B = (B.astype(np.float32) - 128) / 128
        L = L_enhanced.astype(np.float32) / 255.0

        lab_combined = np.stack([L, A, B], axis=0)  # shape: [3, H, W]
        lab_resized = torch.tensor(cv2.resize(lab_combined.transpose(1, 2, 0), (512, 512)).transpose(2, 0, 1), dtype=torch.float32)

        image_tensor = lab_resized  # shape: [3, 512, 512]

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from 'rgb', 'green', or 'lab'.")

    return image_tensor





def load_model(model_path, in_channels=3, class_size=1):
    '''
    Load a pretrained U-Net model.
    
    Args:
        model_path: Path to the model weights file
        in_channels: Number of input channels
        class_size: Number of output classes
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=class_size
    ).to(device)
    
    # Load model weights
    if isinstance(model_path, str):
        # If it's a file path
        state_dict = torch.load(model_path, map_location=device)
    else:
        # If it's a file-like object
        import io
        model_bytes = model_path.read()
        buffer = io.BytesIO(model_bytes)
        state_dict = torch.load(buffer, map_location=device)
        
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    return model





def get_mask(image_tensor, model):

    '''
    Run inference on a single image tensor and return predicted binary mask.
    '''

    device = next(model.parameters()).device
    image_tensor = image_tensor.unsqueeze(0).to(device)  # shape: [1, C, H, W]

    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # [H, W]
        binary_mask = (pred_mask > threshold).astype(np.uint8) * 255

    return binary_mask




def get_exudates_area(mask):
    '''
    Calculate exudates area from binary mask.
    '''
    return np.sum(mask > 0)

def predict_exudates(img_input, model_path=None, model=None):
    '''
    Given an image path or PIL Image, performs inference to calculate exudate area and returns:
    - original image
    - binary mask
    - exudate area (in pixels)
    
    Args:
        img_input: Can be either a file path (str) or a PIL Image object
        model_path: Path to the model weights file (required if model is not provided)
        model: Pre-loaded model (optional, if not provided, will load from model_path)
    '''
    if model is None and model_path is None:
        raise ValueError("Either model or model_path must be provided")
    temp_path = None
    try:
        # Handle both file path and PIL Image inputs
        if isinstance(img_input, str):
            # Input is a file path
            image_tensor = load_image(img_input, mode='rgb')
            # Convert tensor to numpy for consistency in return type
            original_image = np.transpose(image_tensor.numpy(), (1, 2, 0))
        elif hasattr(img_input, 'save'):  # Check if it's a PIL Image
            # Convert PIL Image to tensor
            transform = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor()
            ])
            image_tensor = transform(img_input)
            # Convert tensor to numpy for consistency in return type
            original_image = np.transpose(image_tensor.numpy(), (1, 2, 0))
            # Save to temp file for load_image
            temp_path = "temp_exudate_input.jpg"
            img_input.save(temp_path)
            img_input = temp_path
        else:
            raise ValueError("Input must be either a file path (str) or a PIL Image")

        # Load model if not provided
        if model is None:
            model = load_model(model_path, in_channels=3)
            
        # Get prediction
        binary_mask = get_mask(image_tensor, model)
        area = get_exudates_area(binary_mask)

        return original_image, binary_mask, area
        
    finally:
        # Clean up temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def segment_exudates(model, image):
    """
    Wrapper function for Streamlit app: accepts PIL image and returns predicted binary mask (PIL.Image).
    """
    # Convert to tensor
    image_tensor = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])(image)

    # Predict mask
    mask_np = get_mask(image_tensor, model)  # returns numpy array [H, W]
    
    # Convert to PIL image
    mask_pil = Image.fromarray(mask_np).convert("L")

    return mask_pil
