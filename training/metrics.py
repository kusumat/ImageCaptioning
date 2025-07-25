from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import torch


def compute_metrics(model, dataloader, tokenizer, device):
    model.eval()
    predictions = []
    references = []
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)

            for pred, ref in zip(predicted_ids, batch['caption']):
                pred_tokens = tokenizer.decode(pred, skip_special_tokens=True).strip()
                predictions.append(pred_tokens)
                references.append(ref)

                pred_embed = logits.mean(dim=1).cpu().numpy()
                embeddings.append(pred_embed)

    # BLEU
    bleu = sum(sentence_bleu([ref.split()], pred.split()) for ref, pred in zip(references, predictions)) / len(predictions)

    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True)

    # METEOR
    meteor = sum(meteor_score([ref], pred) for ref, pred in zip(references, predictions)) / len(predictions)

    # Cosine
    cosine = cosine_similarity(torch.tensor(embeddings).squeeze(1)).mean().item()

    return {
        "bleu": bleu,
        "rouge": rouge_scores["rouge-l"]["f"],
        "meteor": meteor,
        "cosine": cosine
    }
