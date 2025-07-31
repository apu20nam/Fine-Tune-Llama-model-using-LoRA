import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from datasets import load_dataset
import pandas as pd

from unsloth import FastLanguageModel
import torch
import wandb
wandb.init(project="Llama3_Telecom_Finetuning", name="Llama3-2_3B_SFT_Run2", config={"epochs": 2, "batch_size": 32})
max_seq_length = 4000 
dtype = None 
load_in_4bit = False 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/data4/fine_tunning/llama3_2_3B_telecom_pretrain", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    
   
)

model = FastLanguageModel.get_peft_model(
    model,
    r =128, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 256,
    lora_dropout = 0, 
    bias = "none",   
    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None,
)


def convert_question_answer_to_llama_conversation(samples):
    
    questions = samples['Question']
    answers = samples['Answer']

    
    instruction_prompt_template = '''
        You are an expert telecom research assistant with deep knowledge of telecommunications, including 5G, RAN, Wireless Communications, and Telco Standards such as 3GPP, ITU, and ETSI.  
        Your primary role is to provide precise, well-reasoned, and contextually accurate answers to user queries. You must analyze complex telecom concepts, apply logical reasoning, and explain solutions step by step.  
        When answering questions, ensure clarity by breaking down technical terms, justifying conclusions with relevant standards and frameworks, and offering practical insights for real-world telecom applications.  
        Use structured reasoning, compare alternatives where applicable, and align responses with industry best practices.
    '''

    
    processed_texts = []
    processed_messages = []
    processed_answers = []

    for question, answer in zip(questions, answers):
        
        if not answer:
            answer = "No answer provided."

       
        messages = [
            {"role": "system", "content": instruction_prompt_template},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        
        sample_conversation = tokenizer.apply_chat_template(messages, tokenize=False)
        
        processed_texts.append(sample_conversation)
        processed_messages.append(messages)
        processed_answers.append(answer)

    
    return {
        "text": processed_texts, 
        "messages": processed_messages, 
        "answer": processed_answers
    }



data1=pd.read_csv('/data4/fine_tunning/dataset/training_data_V1.csv')
data2=pd.read_csv('/data4/fine_tunning/dataset/TRAIN_MCQ_Questions_Answers.csv')
data3=pd.read_csv('/data4/fine_tunning/mmlu_samples_data.csv', dtype={'Answer': str})
concatenated = pd.concat([data1, data2,data3])
dataset= concatenated.sample(frac=1).reset_index(drop=True)
dataset=Dataset.from_pandas(dataset)
dataset = dataset.map( convert_question_answer_to_llama_conversation, batched=True)

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_steps = 1000, 
        learning_rate = 5e-6,  
        weight_decay = 0.05,  
        max_grad_norm = 1.0,  
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,  
        optim = "adamw_torch",
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "Output3_2_3B",
        report_to = "wandb",
        save_strategy = "steps",
        num_train_epochs = 2, 
    ),
)


trainer_stats = trainer.train()

print("model has trained ")
model.save_pretrained("llama3.2:3b") 
tokenizer.save_pretrained("llama3.2:3b")

