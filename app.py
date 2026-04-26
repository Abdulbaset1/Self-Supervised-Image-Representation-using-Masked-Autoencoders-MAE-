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

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Force CPU usage
device = torch.device("cpu")

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

# Define the complete model architecture (not relying on traced model)
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
        self.patch_size = PATCH
        self.num_patches = NUM_PATCHES
        
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
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - L, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x_ = x_ + self.pos_embed_dec
        
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        pred = self.head(x_)
        
        return pred, patches, mask
    
    def patchify(self, imgs):
        B, C, H, W = imgs.shape
        patches = imgs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.reshape(B, -1, self.patch_size * self.patch_size * C)
        return patches
    
    def random_masking(self, x, mask_ratio=0.75):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

def unpatchify(patches, patch_size=PATCH):
    B, N, D = patches.shape
    p = patch_size
    h = w = int(N ** 0.5)
    patches = patches.reshape(B, h, w, 3, p, p)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    imgs = patches.reshape(B, 3, h * p, w * p)
    return imgs

@st.cache_resource
def load_model():
    """Load the MAE model weights from checkpoint"""
    model_path = "mae_deployment.pt"
    state_dict_path = "mae_state_dict.pt"
    
    # First try to download the checkpoint
    if not os.path.exists(model_path) and not os.path.exists(state_dict_path):
        with st.spinner("Downloading model (this may take a minute)..."):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(MODEL_URL, headers=headers, stream=True)
                response.raise_for_status()
                
                # Show progress
                progress_bar = st.progress(0)
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress_bar.progress(min(downloaded / total_size, 1.0))
                
                progress_bar.empty()
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Download failed: {str(e)}")
                return None
    
    # Load model
    try:
        # Create model instance
        model = MAE().to(device)
        
        # Try loading as state dict first (if it's a checkpoint)
        if os.path.exists(state_dict_path):
            checkpoint = torch.load(state_dict_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        elif os.path.exists(model_path):
            # Try loading as traced model
            try:
                traced_model = torch.jit.load(model_path, map_location=device)
                # Extract state dict from traced model (if possible)
                st.info("Loading traced model...")
                model = traced_model
            except:
                # Try as regular checkpoint
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint)
                else:
                    st.warning("Using model as is...")
                    model = checkpoint
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.info("Attempting to create a new model with random weights for demonstration...")
        # Fallback to random model for testing
        model = MAE().to(device)
        model.eval()
        return model

def preprocess_image(image):
    """Preprocess the input image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0).to(device)

def main():
    st.title("🎨 Masked Autoencoder (MAE) Image Reconstruction")
    st.markdown("""
    <div style='text-align: center'>
    <p>Reconstruct images from masked patches using Vision Transformer based Masked Autoencoder</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        mask_ratio = st.slider(
            "Mask Ratio", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.75, 
            step=0.05,
            help="Percentage of image patches to mask (higher = more challenging reconstruction)"
        )
        
        st.header("📦 Model Status")
        model = load_model()
        
        if model is not None:
            st.success("✅ Model loaded and ready!")
            st.info(f"Running on: CPU")
        else:
            st.error("❌ Model failed to load")
            st.stop()
    
    # Main content
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload any image (will be resized to 224x224)"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Original Image")
            st.image(image, use_container_width=True)
        
        # Process button
        if st.button("🚀 Reconstruct Image", type="primary", use_container_width=True):
            with st.spinner("🧠 Processing image through MAE model..."):
                try:
                    # Preprocess image
                    img_tensor = preprocess_image(image)
                    
                    # Run model
                    with torch.no_grad():
                        if isinstance(model, torch.jit.ScriptModule):
                            # For traced models
                            pred, patches, mask = model(img_tensor)
                        else:
                            # For regular models
                            pred, patches, mask = model(img_tensor)
                    
                    # Reconstruct images
                    reconstructed = unpatchify(pred)[0].cpu()
                    
                    # Create masked input visualization
                    masked_input = patches.clone()
                    masked_input[mask.bool()] = 0
                    masked_img = unpatchify(masked_input)[0].cpu()
                    
                    # Display reconstructed image
                    with col2:
                        st.subheader("🔄 Reconstructed Image")
                        img_display = reconstructed.permute(1, 2, 0).numpy()
                        img_display = np.clip(img_display, 0, 1)
                        st.image(img_display, use_container_width=True)
                    
                    # Show detailed visualizations
                    st.markdown("---")
                    st.subheader("📊 Detailed Analysis")
                    
                    tab1, tab2, tab3 = st.tabs(["Masked Input", "Mask Pattern", "Metrics"])
                    
                    with tab1:
                        st.caption("Image with masked patches (black = masked)")
                        masked_display = masked_img.permute(1, 2, 0).numpy()
                        masked_display = np.clip(masked_display, 0, 1)
                        st.image(masked_display, use_container_width=True)
                    
                    with tab2:
                        st.caption("Visualization of which patches were masked")
                        mask_vis = mask[0].cpu().reshape(14, 14).numpy()
                        fig, ax = plt.subplots(figsize=(8, 8))
                        im = ax.imshow(mask_vis, cmap='RdYlBu_r', vmin=0, vmax=1)
                        ax.set_title(f"Mask Pattern (Mask Ratio: {mask_ratio:.0%})", fontsize=14)
                        ax.set_xticks(range(0, 14, 2))
                        ax.set_yticks(range(0, 14, 2))
                        ax.set_xlabel("Patch X coordinate")
                        ax.set_ylabel("Patch Y coordinate")
                        plt.colorbar(im, ax=ax, label="Masked (1) / Visible (0)")
                        st.pyplot(fig)
                        plt.close()
                    
                    with tab3:
                        # Calculate metrics
                        mse = float(torch.mean((reconstructed - img_tensor[0].cpu()) ** 2).item())
                        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                        
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric(
                                "Mean Squared Error (MSE)", 
                                f"{mse:.6f}",
                                help="Lower is better - measures pixel-wise difference"
                            )
                        with col_m2:
                            st.metric(
                                "Peak Signal-to-Noise Ratio (PSNR)", 
                                f"{psnr:.2f} dB",
                                help="Higher is better - measures reconstruction quality"
                            )
                        
                        st.info(f"""
                        **Interpretation:**
                        - MSE < 0.01: Excellent reconstruction
                        - MSE 0.01-0.05: Good reconstruction
                        - MSE > 0.05: Poor reconstruction
                        
                        PSNR > 30 dB: Excellent quality
                        PSNR 20-30 dB: Good quality
                        PSNR < 20 dB: Poor quality
                        """)
                    
                except Exception as e:
                    st.error(f"❌ Error during reconstruction: {str(e)}")
                    st.exception(e)
    
    else:
        # Show placeholder when no image uploaded
        st.info("👈 **Get Started:** Upload an image to begin!")
        
        with st.expander("ℹ️ **About This Model**", expanded=True):
            st.markdown("""
            ### 🧠 Masked Autoencoder (MAE)
            
            This model implements **Masked Autoencoder** for self-supervised image representation learning.
            
            **Architecture Details:**
            - **Encoder:** Vision Transformer Base (12 layers, 768 dimensions, 12 attention heads)
            - **Decoder:** Vision Transformer Small (12 layers, 384 dimensions, 6 attention heads)
            - **Patch Size:** 16×16 pixels
            - **Input Size:** 224×224 pixels (14×14 patches)
            - **Training Data:** Tiny ImageNet (200 classes)
            
            **How It Works:**
            1. 📸 Image is divided into 196 patches
            2. 🎭 Random patches are masked (hidden from the model)
            3. 🔍 Encoder processes only visible patches
            4. 🎨 Decoder reconstructs the full image from representations
            5. 📈 Model learns to understand visual concepts without labels
            
            **Try It Out:**
            - Upload any image (will be resized to 224×224)
            - Adjust mask ratio to control difficulty
            - See how well the model reconstructs masked regions
            """)

if __name__ == "__main__":
    main()
