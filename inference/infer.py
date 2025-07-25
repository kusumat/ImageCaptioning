import torch
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer
from models.vit_gpt2 import ViTGPT2CaptioningModel
from utils.device import get_device

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def generate_caption(image_path, model_path, tokenizer_path="gpt2"):
    device = get_device()
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = ViTGPT2CaptioningModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_tensor = load_image(image_path).to(device)
    input_ids = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        for _ in range(40):  # max length
            outputs = model(pixel_values=image_tensor, input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.ones_like(input_ids)

            if tokenizer.decode(next_token.item()) in ["<|endoftext|>", "."]:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
