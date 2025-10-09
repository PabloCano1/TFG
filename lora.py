# %%
from enum import Enum
from functools import partial
import pandas as pd
import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType
import os

seed = 42
set_seed(seed)

# Put your HF Token here
os.environ['HF_TOKEN']="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" # the token should have write access

model_name = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(sample):
    #load prompt template
    with open('./data/test1.prompt','r') as f:
        prompt_template = f.read()
    #fill the template
    prompt = prompt_template.format(input=sample["question"], output='<start>' + sample["text"]+'<end>')
    return {"text": prompt}

# Corrección: cargar CSV local correctamente
dataset = load_dataset('csv', data_files='./data/Discourse_UWO_PAR_100.csv')
dataset = dataset.map(preprocess, remove_columns=["question", "text",'filename','role','lenP','lenI'])
dataset = dataset["train"].train_test_split(0.1)

# %%
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             attn_implementation='eager',
                                             device_map="cuda",dtype=torch.bfloat16)
model.config.use_cache = False

# %%
rank_dimension = 4
lora_alpha = 32
lora_dropout = 0.1

peft_config = LoraConfig(r=rank_dimension,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         target_modules=[
                             "q_proj", "k_proj", "v_proj",
                             "o_proj", "gate_proj", "up_proj",
                             "down_proj"
                             ],
                         task_type=TaskType.CAUSAL_LM)

# %%
username = "PabloCano1" # replace with your Hugging Face username
output_dir = "modelo1_LoRA_bueno"
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 1
learning_rate = 1e-4

num_train_epochs=10
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_seq_length = 500
# max_grad_norm= 1 ##########
training_arguments = SFTConfig(
    output_dir=output_dir,
    hub_private_repo=False,
    push_to_hub=True,


    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="no",
    eval_strategy="epoch",
    # logging_steps=logging_steps,
    learning_rate=learning_rate,
    # max_grad_norm=max_grad_norm,
    weight_decay=0.1,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing=False,
    max_length=None,
)

# %%
torch.cuda.empty_cache()

# %%
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

# %%
# Usar el modelo fusionado para predecir la respuesta al prompt de antes
from transformers import AutoTokenizer
modelo_unico = trainer.model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("PabloCano1/tercer_modelo")

input_text = """### Instruction:\nRespond as a patient without schizophrenia to the psychologist:\n\n### Input:can you tell me a bit about yourself .\n\n### Expected Response:"""

inputs = tokenizer(input_text, return_tensors="pt").to(modelo_unico.device)
with torch.no_grad():
    outputs = modelo_unico.generate(
        **inputs,
        # max_new_tokens=100,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response[len(input_text):].strip())

# %%
# Fusionar LoRA con el modelo base y subir el modelo único a Hugging Face Hub en un nuevo directorio
modelo_unico = trainer.model.merge_and_unload()
modelo_unico.push_to_hub("PabloCano1/modelo_n3")
tokenizer.push_to_hub("PabloCano1/modelo_n3")