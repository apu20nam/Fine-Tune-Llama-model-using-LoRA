from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import torch
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


os.environ["WANDB_API_KEY"] = "22f9ed3360660835af1a833bb5bd8b2db04bd608"
wandb.login()
wandb.init(
    project="qwen-finetuning",
    name="qwen2.5-0.5b-metamath_task1",
    config={
        "model_name": "Qwen/Qwen2.5-0.5B",
        "dataset": "meta-math/MetaMathQA-40K",
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 2e-5,
        "batch_size": 8,
        "epochs": 1,
        "max_length": 512
    }
)

model_name = "Qwen/Qwen2.5-0.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

rank = 32
alpha = 64
lora_config = LoraConfig(
    r=rank,
    lora_alpha=alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

formatted_data = []

dataset_gen = load_dataset("meta-math/MetaMathQA-40K")

for row in dataset_gen['train']:
    conversation = [
        {"role": "user", "content": row["query"]},
        {"role": "assistant", "content": row["response"]}
    ]
    formatted_data.append({"messages": conversation})

dataset = Dataset.from_list(formatted_data)

def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return example

processed_train_dataset = dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    desc="Applying chat template to train_sft",
)

dataset = processed_train_dataset.map(
    lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=512),
    batched=True,
    remove_columns=['text']
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    per_device_train_batch_size=8,  
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    gradient_accumulation_steps=1,  
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=500,  
    output_dir="./qwen_finetuned",
    fp16=True,
    save_total_limit=2,
    report_to="wandb",  
    dataloader_drop_last=True,  
    label_names=["labels"],
    #logging_dir="./logs",  
    save_strategy="steps",
    #evaluation_strategy="no",  # No evaluation dataset
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)


print("Starting training...")
trainer.train()


print("Saving the final model...")
output_dir = "./qwen_finetuned_final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)



print(f"Model saved to: {output_dir}")



wandb.finish()

print("Training completed and model saved!")