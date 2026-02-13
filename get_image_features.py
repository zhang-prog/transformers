import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def init_model_and_processor(model_path):
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    return model, processor

def get_image_features(model, processor, image_url):
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    inputs = processor(images=image, text="").to(model.device)
    image_feature_gap, image_feature_gmp = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"]) # torch.Size([1, 576]) bf16
    image_feature_gap = image_feature_gap.to(torch.float32).detach().cpu().numpy() # torch.tensor(bf16) -> numpy.ndarray(fp32)
    image_feature_gmp = image_feature_gmp.to(torch.float32).detach().cpu().numpy() # torch.tensor(bf16) -> numpy.ndarray(fp32)

    return image_feature_gap, image_feature_gmp

if __name__ == "__main__":
    model_path = "PaddlePaddle/PaddleOCR-VL-1.5" # model repo id or model directory
    model, processor = init_model_and_processor(model_path)

    image_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg"
    image_feature_gap, image_feature_gmp = get_image_features(model, processor, image_url)
    print(image_feature_gap)
    print(image_feature_gmp)
