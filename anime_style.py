# Install dependencies (if not already installed)
# Run this only once or skip if already installed
# !pip install torch torchvision opencv-python matplotlib

# Import libraries
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load pretrained AnimeGANv2 model via Torch Hub
model = torch.hub.load(
    'bryandlee/animegan2-pytorch:main',
    'generator',
    pretrained='face_paint_512_v2'
).eval()

image_path = 'models/your_face_image.jpeg'  # 🔁 Replace with your image file name
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([

transforms.CenterCrop(min(image.size)),    
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0)

# Run inference (no gradients needed)
with torch.no_grad():
    output = model(input_tensor)[0]

# Convert output tensor to image format
output_image = output.permute(1, 2, 0).numpy()
output_image = (output_image * 255).astype(np.uint8)

# Save the output image
output_filename = "output_face_paint.jpg"
cv2.imwrite(output_filename, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

print(f"✅ Anime-style image saved as: {output_filename}")

# (Optional) Show result in a window or notebook
plt.imshow(output_image)
plt.axis('off')
plt.title("AnimeGANv2 Output")
plt.show()
input_dir = "../input_images"
output_dir = "../output_images"
def compare_images(original_path, anime_path, save_path="comparison.jpg"):
    orig = Image.open(original_path).convert("RGB")
    anime = Image.open(anime_path).convert("RGB")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(orig)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(anime)
    ax[1].set_title("Anime Output")
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Comparison saved as {save_path}")