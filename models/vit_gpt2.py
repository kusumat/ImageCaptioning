import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel

class ViTGPT2CaptioningModel(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224', gpt2_model_name='gpt2'):
        super(ViTGPT2CaptioningModel, self).__init__()

        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

        self.vit_proj = nn.Linear(self.vit.config.hidden_size, self.gpt2.config.n_embd)
        self.gpt2.resize_token_embeddings(self.gpt2.config.vocab_size)

    def forward(self, pixel_values, input_ids, attention_mask):
        # Get image embedding from ViT
        vit_outputs = self.vit(pixel_values=pixel_values)
        img_embeds = vit_outputs.last_hidden_state[:, 0]  # CLS token

        # Project ViT CLS token to GPT2 hidden size
        img_embeds_proj = self.vit_proj(img_embeds).unsqueeze(1)

        # Embed input ids with GPT2 embeddings
        input_embeds = self.gpt2.transformer.wte(input_ids)

        # Concatenate image embedding to the front
        concat_embeds = torch.cat([img_embeds_proj, input_embeds[:, 1:, :]], dim=1)

        outputs = self.gpt2(inputs_embeds=concat_embeds, attention_mask=attention_mask)
        return outputs
