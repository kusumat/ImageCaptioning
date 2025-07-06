Downloading pre-trained BERTScore model files
makefile
Copy
Edit
tokenizer_config.json: 100%|████...|
config.json: 100%|████...|
vocab.json: 100%|████...|
merges.txt: 100%|████...|
tokenizer.json: 100%|████...|
model.safetensors: 100%|████...|
These progress bars show that bert_score is downloading a pre-trained roberta-large model:

bert_score uses contextual embeddings (from RoBERTa) to compare predicted captions vs ground-truth.

These files are cached locally, so next time it will be faster.


✅ Normal — this happens the first time you run bert_score.

2️⃣ Roberta warning: “Some weights... newly initialized”
pgsql
Copy
Edit
Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a downstream task to be able to use it for predictions and inference.
➡️ This is just a standard transformers warning:

The RoBERTa encoder has an optional pooling layer (pooler) used for classification tasks.

bert_score doesn’t need it — so it’s uninitialized.

✅ Totally fine — ignore it for bert_score!

3️⃣ Your actual scores
yaml
Copy
Edit
BLEU-1: 0.4370 | BLEU-4: 0.0823 | BERTScore F1: 0.1968
This is the interesting part:

BLEU-1 = 0.4370 → Unigram overlap is ~43%.
→ Basic word matches are working.

BLEU-4 = 0.0823 → Four-gram overlap is low (~8%).
→ Your model generates basic word-level phrases but struggles with longer coherent sequences.

BERTScore F1 = 0.1968 → Semantic similarity is ~0.20 (low-ish).
→ Indicates generated captions are only somewhat similar in meaning.

✅ Interpretation
These are reasonable for an early prototype.

BLEU-1 > 0.4 means your vocabulary & tokenization pipeline is working.

BLEU-4 low? → Typical in early training, since predicting long correct n-grams is harder.

BERTScore helps you see if synonyms or paraphrases are better than raw word overlap.
