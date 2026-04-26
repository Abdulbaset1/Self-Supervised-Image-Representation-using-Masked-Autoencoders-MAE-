import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import io
import sys

# Disable matplotlib interactive mode to avoid issues
plt.switch_backend('Agg')

# Configuration
PATCH = 16
NUM_PATCHES = (224 // PATCH) ** 2
MODEL_URL = "https://github.com/Abdulbaset1/Self-Supervised-Image-Representation-using-Masked-Autoencoders-MAE-/releases/download/v1/mae_deployment.pt"

# Page configuration
st.set_page_config(
    page_title="MAE Image Reconstruction",
    page_icon="🎨",
    layout="wide"
)

# Define the same model architecture
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        return x + h

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # encoder (ViT Base)
        self.pos_embed_enc = nn.Parameter(torch.randn(1, NUM_PATCHES, 768))
        
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(768, 12) for _ in range(12)]
        )
        
        self.encoder_norm = nn.LayerNorm(768)
        
        # decoder projection
        self.proj = nn.Linear(768, 384)
        
        # decoder (ViT Small)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 384))
        
        self.pos_embed_dec = nn.Parameter(torch.randn(1, NUM_PATCHES, 384))
        
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(384, 6) for _ in range(12)]
        )
        
        self.decoder_norm = nn.LayerNorm(384)
        
        self.head = nn.Linear(384, PATCH * PATCH * 3)

    def forward(self, imgs):
        patches = self.patchify(imgs)
        visible, mask, ids_restore = self.random_masking(patches)
        
        # encoder
        x = visible + self.pos_embed_enc[:, :visible.shape[1]]
        
        for blk in self.encoder_blocks:
            x = blk(x)
        
        x = self.encoder_norm(x)
        
        # projection to decoder
        x = self.proj(x)
        
        B, L, D = x.shape
        
        mask_tokens = self.mask_token.repeat(
            B, ids_restore.shape[1] - L, 1
        )
        
        x_ = torch.cat([x, mask_tokens], dim=1)
        
        x_ = torch.gather(
            x_,
            1,
            ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )
        
        x_ = x_ + self.pos_embed_dec
        
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        
        x_ = self.decoder_norm(x_)
        
        pred = self.head(x_)
        
        return pred, patches, mask
    
    def patchify(self, imgs):
        B, C, H, W = imgs.shape
        patches = imgs.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
        patches = patches.contiguous().view(B, C, -1, PATCH, PATCH)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.reshape(B, -1, PATCH * PATCH * C)
        return patches
    
    def unpatchify(self, patches):
        B, N, D = patches.shape
        p = PATCH
        h = w = int(N ** 0.5)
        
        patches = patches.reshape(B, h, w, 3, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        imgs = patches.reshape(B, 3, h * p, w * p)
        
        return imgs
    
    def random_masking(self, x, mask_ratio=0.75):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        
        x_visible = torch.gather(
            x, 1,
            ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        return x_visible, mask, ids_restore

# Global functions for patchify and unpatchify
def patchify(imgs):
    B, C, H, W = imgs.shape
    patches = imgs.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
    patches = patches.contiguous().view(B, C, -1, PATCH, PATCH)
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.reshape(B, -1, PATCH * PATCH * C)
    return patches

def unpatchify(patches):
    B, N, D = patches.shape
    p = PATCH
    h = w = int(N ** 0.5)
    
    patches = patches.reshape(B, h, w, 3, p, p)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    imgs = patches.reshape(B, 3, h * p, w * p)
    
    return imgs

def random_masking(x, mask_ratio=0.75):
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    
    noise = torch.rand(B, N, device=x.device)
    
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    ids_keep = ids_shuffle[:, :len_keep]
    
    x_visible = torch.gather(
        x, 1,
        ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )
    
    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    
    return x_visible, mask, ids_restore

@st.cache_resource
def load_model():
    """Load the MAE model from local file or download from GitHub"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = "models/mae_deployment.pt"
    
    # Download model if not exists
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from GitHub release..."):
            try:
                # Add headers to avoid rate limiting
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(MODEL_URL, headers=headers, stream=True)
                response.raise_for_status()
                
                # Download with progress bar
                progress_bar = st.progress(0)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(model_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = min(downloaded / total_size, 1.0)
                            progress_bar.progress(progress)
                
                progress_bar.empty()
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")
                return None, None
    
    # Load model
    try:
        # Load the traced model
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess the input image for the model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def reconstruct_image(model, img_tensor, device, mask_ratio=0.75):
    """Reconstruct the image using the MAE model"""
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        
        # Forward pass
        pred, patches, mask = model(img_tensor)
        
        # Reconstruct both masked and complete images
        reconstructed = unpatchify(pred)
        
        # Create masked input visualization
        masked_input = patches.clone()
        masked_input[mask.bool()] = 0
        masked_imgs = unpatchify(masked_input)
        
        return reconstructed[0].cpu(), masked_imgs[0].cpu(), mask[0].cpu()

def main():
    st.title("🖼️ Masked Autoencoder (MAE) Image Reconstruction")
    st.markdown("""
    This app demonstrates a Masked Autoencoder trained on Tiny ImageNet. 
    Upload an image, and the model will reconstruct it from masked patches!
    """)
    
    # Sidebar controls
    st.sidebar.header("Settings")
    mask_ratio = st.sidebar.slider(
        "Mask Ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.75,
        step=0.05,
        help="Percentage of image patches to mask"
    )
    
    # Load model
    st.sidebar.header("Model Status")
    model, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check your internet connection and try again.")
        return
    
    st.sidebar.success(f"✅ Model loaded successfully")
    
    # Image upload
    st.header("📤 Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Process button
        if st.button("🎨 Reconstruct Image", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Preprocess
                    img_tensor = preprocess_image(image)
                    
                    # Reconstruct
                    reconstructed, masked_input, mask = reconstruct_image(
                        model, img_tensor, device, mask_ratio
                    )
                    
                    # Convert tensors to displayable format
                    def tensor_to_image(tensor):
                        img = tensor.permute(1, 2, 0).numpy()
                        img = np.clip(img, 0, 1)
                        return img
                    
                    masked_img_display = tensor_to_image(masked_input)
                    reconstructed_img_display = tensor_to_image(reconstructed)
                    
                    # Display results
                    with col2:
                        st.subheader("Reconstructed Image")
                        st.image(reconstructed_img_display, use_container_width=True)
                    
                    # Additional visualizations
                    st.header("📊 Visualization Results")
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.subheader("Masked Input")
                        st.image(masked_img_display, use_container_width=True)
                        st.caption(f"Mask Ratio: {mask_ratio:.0%}")
                    
                    with col4:
                        # Display mask pattern
                        st.subheader("Mask Pattern")
                        mask_vis = mask.reshape(14, 14).numpy()  # 224/16 = 14
                        fig, ax = plt.subplots(figsize=(6, 6))
                        im = ax.imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
                        ax.set_title("Masked Patches (White = Masked)")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Metrics
                    st.header("📈 Reconstruction Metrics")
                    mse = float(torch.mean((reconstructed - img_tensor[0].cpu()) ** 2).item())
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("MSE (Lower is better)", f"{mse:.4f}")
                    with metric_col2:
                        st.metric("PSNR (Higher is better)", f"{psnr:.2f} dB")
                        
                except Exception as e:
                    st.error(f"Error during reconstruction: {str(e)}")
                    st.exception(e)
    
    else:
        # Show example when no image is uploaded
        st.info("👈 Please upload an image to get started!")
        
        # Display sample information
        with st.expander("ℹ️ About this Model"):
            st.markdown("""
            ### Model Architecture
            - **Encoder**: Vision Transformer Base (12 layers, 768 dim, 12 heads)
            - **Decoder**: Vision Transformer Small (12 layers, 384 dim, 6 heads)
            - **Patch Size**: 16x16 pixels
            - **Input Size**: 224x224 pixels
            - **Training Dataset**: Tiny ImageNet (200 classes)
            
            ### How it works
            1. The image is divided into 196 patches (14x14 grid)
            2. A random subset of patches are masked (hidden)
            3. The encoder processes only the visible patches
            4. The decoder reconstructs the full image including masked patches
            5. The model learns to understand image structure and content
            
            ### Tips for best results
            - Use images with clear subjects and simple backgrounds
            - Square images work best (will be resized to 224x224)
            - Lower mask ratios (0.5-0.6) give better reconstruction
            - Higher mask ratios (0.75-0.85) show the model's predictive capabilities
            """)

if __name__ == "__main__":
    main()
