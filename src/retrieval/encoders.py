"""Frozen image encoders; no ResNet-50 weights are requested."""
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

MODELS = ('resnet18', 'dinov2', 'clip')


def load_encoder(name, device):
    if name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        preprocess = weights.transforms()
        def process(images):
            return torch.stack([preprocess(image) for image in images]).to(device)
        def forward(inputs):
            return model(inputs)
    else:
        repo = 'facebook/dinov2-small' if name == 'dinov2' else 'openai/clip-vit-base-patch32'
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo) if name == 'dinov2' else CLIPVisionModelWithProjection.from_pretrained(repo)
        def process(images):
            return {k: v.to(device) for k, v in processor(images=images, return_tensors='pt').items()}
        def forward(inputs):
            result = model(**inputs)
            return result.last_hidden_state[:, 0] if name == 'dinov2' else result.image_embeds
    model.to(device).eval()
    return model, process, forward


def encode(name, paths, device, batch_size=16):
    model, process, forward = load_encoder(name, device)
    with Image.open(paths[0]) as im:
        warmup = process([im.convert('RGB')])
    with torch.inference_mode():
        forward(warmup)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    chunks = []
    with torch.inference_mode():
        for begin in range(0, len(paths), batch_size):
            images = []
            for path in paths[begin:begin+batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            chunks.append(forward(process(images)).cpu().numpy())
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(chunks).astype('float32'), elapsed
