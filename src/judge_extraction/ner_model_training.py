import ast
import spacy
import pandas as pd
from pathlib import Path
import random
from spacy.training import Example
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load manually annotated data
gold_standard_annotations = pd.read_csv('../../data/processed/ner/judges_gold_standard_annotations.csv.gz', compression='gzip')

# Remove duplicates
gold_standard_annotations = gold_standard_annotations.groupby(by=['text', 'annotations', 'source']
).size().reset_index(name='count')
gold_standard_annotations.drop(columns=['count'], inplace=True)

# Parse annotations into Python objects
gold_standard_annotations['annotations'] = gold_standard_annotations['annotations'].apply(ast.literal_eval)

# Split into training and test sets
train, test = train_test_split(gold_standard_annotations, test_size=0.2, random_state=42)
print(f"Training samples: {len(train)}, Test samples: {len(test)}")

# Download and load spaCy model
spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

# Convert test examples into spaCy format
examples_test = [
    Example.from_dict(nlp.make_doc(row['text']), row['annotations']) for _, row in test.iterrows()
]

# Evaluate the model before training
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

with nlp.disable_pipes(*unaffected_pipes):
    score = nlp.evaluate(examples_test)
    print("Initial evaluation score:", score)

# Enable GPU for training
spacy.prefer_gpu()

# Prepare training data
train_data = train[['text', 'annotations']].apply(lambda row: (row['text'], row['annotations']), axis=1).tolist()

# Training the model
list_losses = []
list_scorer = []

with nlp.disable_pipes(*unaffected_pipes):
    for iteration in range(100):  # Number of iterations
        print(f"Iteration {iteration + 1}/100")
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

        for batch in batches:
            examples = [
                Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch
            ]
            nlp.update(examples, drop=0.3, losses=losses)

        # Evaluate after each iteration
        scores = nlp.evaluate(examples_test)
        list_scorer.append((iteration, scores))
        list_losses.append((iteration, losses.get("ner", 0)))
        print(f"Iteration {iteration + 1} - Loss: {losses.get('ner', 0)}, Score: {scores}")

# Save training losses
losses_df = pd.DataFrame(list_losses, columns=["iteration", "loss"])
losses_df.to_csv('../../data/processed/ner/losses.csv', index=False)

# Final evaluation
with nlp.disable_pipes(*unaffected_pipes):
    final_score = nlp.evaluate(examples_test)
    print("Final evaluation score:", final_score)

# Save the trained model
output_dir = Path('../../data/processed/ner/model')
output_dir.mkdir(parents=True, exist_ok=True)
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")
