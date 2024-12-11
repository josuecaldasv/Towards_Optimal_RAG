from dotenv import load_dotenv
import os
import sys
from datasets import Dataset, DatasetDict
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator, AutoTokenizer
import torch

load_dotenv()

sys.path.append(os.path.abspath('../funcs'))
import functions as fn

TOKEN = os.getenv('HF_TOKEN')
model_name = 'NousResearch/Hermes-3-Llama-3.1-70B'
url = 'https://raw.githubusercontent.com/ICA-PUC/nlp-research/refs/heads/master/knowledge-graph-question-answering/Training%20Data%20-miniKGraph/dataset_miniKGraph.json?token=GHSAT0AAAAAACUJDTTAGHSZFUVNUMFWE4Y6Z2SFIZA'
tokenizer = AutoTokenizer.from_pretrained(model_name)


data = fn.load_ft_dataset(url, TOKEN)
processed_data = fn.process_ft_dataset(data, 0.8)
train_data = fn.convert_ft_dataset_to_columnar_format(processed_data["train"])
test_data = fn.convert_ft_dataset_to_columnar_format(processed_data["test"])

dataset_dict_data = DatasetDict({
    "train": Dataset.from_dict(train_data),
    "test": Dataset.from_dict(test_data)
})

print(dataset_dict_data['train'][20])

tokenized_data = dataset_dict_data.map(
    lambda examples: fn.tokenize_train_features_ft_dataset(examples, tokenizer),
    batched=True,
    remove_columns=dataset_dict_data["train"].column_names
)

print(tokenized_data['train'][20])


train_dataset = tokenized_data["train"]
eval_dataset = tokenized_data["test"]

args = TrainingArguments(
    output_dir="finetune-model",
    eval_strategy="epoch",
    save_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
data_collator = DefaultDataCollator()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("hermes3_fine-tuned")