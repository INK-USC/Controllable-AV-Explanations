# Llama-3 training code derived and modified from https://github.com/shallinan1/StyleRemix
import os
os.environ['TRANSFORMERS_CACHE'] = '../cache/'
os.environ['HF_HOME'] = '../cache/'
os.environ['HUGGINGFACE_HUB_CACHE'] = '../cache/'

from IPython import embed
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import math

from datetime import datetime
time = datetime.now()    
date_time = time.strftime("%m-%d-%Y_%H_%M_%S")

from peft import (
    LoraConfig,
    PeftConfig,
    TaskType,
    PeftModel,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainingArguments
)
import torch
from datasets import Dataset
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from huggingface_hub import login
import json
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import pdb

def main(args):
    output_dir = os.path.join(args.output_dir,date_time)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"\n\t*\tSaving to {output_dir}\n")
    set_seed(args.seed)

    data = pd.read_csv(args.data_path)

    all_src = []
    all_tgt = []
    for k in range(len(data)): 
        all_src.append(data["input"][k])
        all_tgt.append(data["output"][k])    

    '''
    # Split into train and dev set
    train_data, eval_data = train_test_split(
        list(zip(all_src, all_tgt)),
        test_size=0.15, 
        random_state=args.seed)
    
    train_src, train_tgt = zip(*train_data)
    eval_src, eval_tgt = zip(*eval_data)

    train_dataset = Dataset.from_dict({
        "src": train_src,
        "tgt": train_tgt}
        )
    eval_dataset = Dataset.from_dict({
        "src": eval_src,
        "tgt": eval_tgt
    })
    '''
    train_dataset = Dataset.from_dict({
        "src": all_src,
        "tgt": all_tgt
        })

    # Set the huggingface token
    if args.hf_token:
        login(args.hf_token)
    
    # Load in (toy) data
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    print(f"\n\t*\tModels are on {device}\n")

    # Initialize Lora Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        bias = "none")
    # TODO fix bug where the EOS token is not appearing
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              add_eos_token=True)
    if not tokenizer.pad_token: # Set pad token if it doesn't exist
        tokenizer.pad_token = tokenizer.eos_token 
    # model = AutoModelForCausalLM.from_pretrained(args.model)

    """
    Three ways to load peft config:
    * AutoModelForCausalLM.from_pretrained(mmodel, peft_config = peft_config)
    * PeftModel.from_pretrained(model, peft_config)
    * peft.get_peft_model(model, peft_config)
    """
    # peft_model = get_peft_model(model, peft_config)
    # print(f"\n\t*\t{peft_model.print_trainable_parameters()}\n") # Print out the trainable parameters
    # del peft_model # We reinstantiate it during training

    steps = len(all_src)/(args.train_batch_size*torch.cuda.device_count())
    # Save every quarter epoch
    save_steps = math.ceil(steps / args.save_ratio)
    print(f"\n\t*\tSave steps is {save_steps}\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        #per_device_eval_batch_size=args.eval_batch_size,
        report_to="tensorboard",
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        #evaluation_strategy='steps',
        num_train_epochs=args.epochs,
        #eval_steps = save_steps,
        save_steps = save_steps,
        logging_steps = save_steps,
        lr_scheduler_type = 'linear',
        seed = args.seed,
        warmup_ratio = 0.1,
    )

    # def add_eos(batch):
    #     print(list(batch.keys()))
    #     outputs=[]
    #     for b in batch["text"]:
    #         outputs.append(b + " <|end_of_text|>")
    #     return outputs

    # # TODO hotfix remove this add_eos when llama tokenizer fixed
    def formatting_prompts_func(example):
        output_texts = []
        prompt_instr = "### Instruction: Explain if Text1 and Text2 have the same author or not, using a dictionary format."
        prompt_instr_tokens = tokenizer.encode(prompt_instr)
        for i in range(len(example['src'])):
            src = example['src'][i]
            tgt = '\n ### Response: ' + example['tgt'][i] + ' <|end_of_text|>'
            op_tokens = tokenizer.encode(tgt)
            ind = src.index("Text 2")
            src_1 = src[:ind]
            src_2 = src[ind:]
            src_1_tokens = tokenizer.encode(src_1)
            src_2_tokens = tokenizer.encode(src_2)

            max_len = args.max_seq_length
            available_len = max_len - len(prompt_instr_tokens) - len(op_tokens) - 4 # -2 for safety, for any spaces after prompt_instr

            if available_len <= 0: 
                continue # can't use this datapoint with this sequence

            per_src_len = available_len//2
            src_1_new = tokenizer.decode(src_1_tokens[:per_src_len], skip_special_tokens=True)
            src_2_new = tokenizer.decode(src_2_tokens[:per_src_len], skip_special_tokens=True)
            
            text = prompt_instr + " " + src_1_new + src_2_new + tgt
            # text = f"### Instruction: {example['src'][i]}\n ### Response: {example['tgt'][i]} <|end_of_text|>"
            output_texts.append(text)
        return output_texts
    response_template = " ### Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        # dataset_text_field="text",
        formatting_func=formatting_prompts_func,
        packing=False,
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=collator
    )
    print("\n")
    trainer.model.print_trainable_parameters() # Print out the trainable parameters
    assert trainer.train_dataset.num_rows == len(all_src) #len(train_data)
    print(tokenizer.decode(trainer.train_dataset[0]['input_ids']))

    trainer.train()
    # trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument(
        '--output_dir', type=str, default="save_deepspeed/llama3")
    parser.add_argument(
        '--hf_token', type=str, default='')
    parser.add_argument(
        '--lora_r', type=int, default=8)
    parser.add_argument(
        '--lora_alpha', type=int, default=16)
    parser.add_argument(
        '--lora_dropout', type=float, default=0.01)
    parser.add_argument(
        '--data_path', type=str, default="../data/imdb62/prompt5/train_i2ro.csv")
    parser.add_argument(
        '--deepspeed', type=str, default=None)
    parser.add_argument(
        '--seed', type=int, default=0)
    parser.add_argument(
        '--save_ratio', type=int, default=4)
    parser.add_argument(
        '--epochs', type=int, default=1)
    parser.add_argument(
        '--local_rank', type=int, default=-1)
    parser.add_argument(
        '--train_batch_size', type=int, default=4)
    parser.add_argument(
        '--eval_batch_size', type=int, default=4)
    parser.add_argument(
        '--max_seq_length', type=int, default=768)
    main(parser.parse_args())
