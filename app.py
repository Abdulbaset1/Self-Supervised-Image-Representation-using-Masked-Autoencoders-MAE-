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

# Set matplotlib backend to 'Agg' for headless environments
import matplotlib
matplotlib.use('Agg')

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

# Define the model architecture
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
    """Load the MAE model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "mae_deployment.pt"
    
    # Download model if not exists
    if not os.path.exists(model_path):
        with st.spinner("Downloading model (this may take a minute)..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded!")
            except Exception as e:
                st.error(f"Download failed: {str(e)}")
                return None, None
    
    # Load model
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess the input image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0)

def main():
    st.title("🖼️ Masked Autoencoder (MAE) Image Reconstruction")
    st.markdown("Reconstruct images from masked patches using Vision Transformer")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        mask_ratio = st.slider("Mask Ratio", 0.1, 0.9, 0.75, 0.05)
        
        st.header("Model Status")
        model, device = load_model()
        
        if model is not None:
            st.success("✅ Model ready")
        else:
            st.error("❌ Model failed to load")
            return
    
    # Main content
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        
        if st.button("Reconstruct", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Process image
                    img_tensor = preprocess_image(image).to(device)
                    
                    # Run model
                    with torch.no_grad():
                        pred, patches, mask = model(img_tensor)
                    
                    # Reconstruct images
                    reconstructed = unpatchify(pred)[0].cpu()
                    masked_input = patches.clone()
                    masked_input[mask.bool()] = 0
                    masked_img = unpatchify(masked_input)[0].cpu()
                    
                    # Display results
                    with col2:
                        st.subheader("Reconstructed")
                        img_display = reconstructed.permute(1, 2, 0).numpy()
                        img_display = np.clip(img_display, 0, 1)
                        st.image(img_display, use_container_width=True)
                    
                    # Show side-by-side comparison
                    st.subheader("Comparison")
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.caption("Masked Input")
                        masked_display = masked_img.permute(1, 2, 0).numpy()
                        masked_display = np.clip(masked_display, 0, 1)
                        st.image(masked_display, use_container_width=True)
                    
                    with comp_col2:
                        st.caption("Mask Pattern")
                        mask_vis = mask[0].cpu().reshape(14, 14).numpy()
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
                        ax.set_title(f"Mask Ratio: {mask_ratio:.0%}")
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    else:
        st.info("👈 Upload an image to get started!")
        
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Masked Autoencoder (MAE)** 
            - Trained on Tiny ImageNet
            - Vision Transformer Base encoder
            - Reconstructs images from 75% masked patches
            - Demonstrates self-supervised learning
            """)

if __name__ == "__main__":
    main()
