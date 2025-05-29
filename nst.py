import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_bytes, max_size=400, shape=None):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def imshow(tensor):
    # Undo normalization and convert tensor to PIL image
    unloader = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1., 1., 1.]),
        transforms.ToPILImage()
    ])
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    return image

# Load pre-trained VGG19 with correct API (weights argument)
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# Layers for content and style features (using layer indices as strings)
content_layers = ['21']  # conv4_2
style_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in content_layers:
            features['content'] = x
        if name in style_layers:
            features.setdefault('style', []).append(x)
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def run_style_transfer(content_img, style_img, steps=300, style_weight=1e6, content_weight=1):
    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.LBFGS([target])

    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)
    style_grams = [gram_matrix(y) for y in style_features['style']]

    run = [0]
    while run[0] <= steps:
        def closure():
            optimizer.zero_grad()
            target_features = get_features(target, vgg)
            content_loss = torch.mean((target_features['content'] - content_features['content'])**2)

            style_loss = 0
            for ft_y, gm_s in zip(target_features['style'], style_grams):
                gm_y = gram_matrix(ft_y)
                style_loss += torch.mean((gm_y - gm_s)**2) / ft_y.numel()

            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()

            if run[0] % 50 == 0:
                st.write(f"Step {run[0]}: Loss {total_loss.item():.4f}")

            run[0] += 1
            return total_loss

        optimizer.step(closure)

    return target.detach()

# Streamlit UI
st.title("Neural Style Transfer App")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

if content_file and style_file:
    content_bytes = content_file.read()
    style_bytes = style_file.read()

    content_img = load_image(content_bytes)
    style_img = load_image(style_bytes, shape=content_img.shape[-2:])

    st.image([Image.open(io.BytesIO(content_bytes)), Image.open(io.BytesIO(style_bytes))],
             caption=["Content Image", "Style Image"], width=300)

    if st.button("Run Style Transfer"):
        with st.spinner("Processing... this may take a minute"):
            output = run_style_transfer(content_img, style_img)
            output_image = imshow(output)
            st.image(output_image, caption="Output Image", use_column_width=True)
