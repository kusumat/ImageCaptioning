import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CaptionDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer, max_length=40):
        self.data = pd.read_csv(csv_path)
        self.data.dropna(inplace=True)
        self.data = self.data[~self.data['comment'].str.contains(",,,")]
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.data.iloc[idx]['image_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.data.iloc[idx]['comment'].strip().replace(",", "")
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()

        return {
            'pixel_values': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption
        }
