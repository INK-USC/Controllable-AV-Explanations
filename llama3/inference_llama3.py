# evaluate causal model
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import pandas as pd
import copy
from tqdm import tqdm 

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    #is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import pdb
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--model_name', type=str, default='meta-llama/Meta-Llama-3-8B',
        help='model name for tokenizer')
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='trained model path')
    parser.add_argument(
        '--do_val', type=int, default=1, help='0/1 whether to do val eval')
    parser.add_argument(
        '--do_test', type=int, default=1, help='0/1 whether to do test eval')
    parser.add_argument(
        '--prefix', type=str, default="", help='prefix for cross domain eval')
    parser.add_argument(
        '--dataset-val', type=str, default='../data/imdb62/prompt5/val_i2ro.csv',
        help='CSV file containing val data.')
    parser.add_argument(
        '--dataset-test', type=str, default='../data/imdb62/prompt5/test_i2ro.csv',
        help='CSV file containing test data.')
    parser.add_argument(
        '--top_p', type=float, default=0.95, help='hyperparameter for nucleus sampling')
    parser.add_argument(
        '--top_k', type=float, default=50, help='hyperparameter for nucleus sampling')
    parser.add_argument(
        '--do_sample', action='store_false', default=True, help='whether to use sampling to decode')
    parser.add_argument(
        '--hf_auth_token', type=str, default='',
        help='auth token for hugging-face')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=args.hf_auth_token)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    if not tokenizer.pad_token: # Set pad token if it doesn't exist
        tokenizer.pad_token = tokenizer.eos_token 

    num_gpus = torch.cuda.device_count()
    print("num_gpus detected:", num_gpus)
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    # instructions
    batch_size = 1 
    prompt_instr = "Explain if Text1 and Text2 have the same author or not, using a dictionary format. "

    # running on validation
    if args.do_val:
        val_data = pd.read_csv(args.dataset_val)
        val_processed = []
        for i in range(len(val_data)):
            ip1 = prompt_instr + val_data["input"][i]
            ip = f"### Instruction: {ip1}\n ### Response: "
            val_processed.append(ip)

        steps = int(len(val_processed)//batch_size) + 1
        val_gen_text = []
        for i in tqdm(range(steps)):
            if i%100==0:
                print(i, steps)
            try:
                batch = val_processed[i*batch_size:(i+1)*batch_size]
            except:
                batch = val_processed[i*batch_size:]
            if len(batch) == 0:
                continue
            #encoded_input = tokenizer(batch, return_tensors='pt', padding='max_length',
            #                            truncation=True, max_length=512, return_token_type_ids=False).to(device)
            encoded_input = tokenizer(batch, return_tensors='pt', return_token_type_ids=False).to(device)
            gen_tokens = model.generate(**encoded_input, max_new_tokens=1024, do_sample=True, 
                                         top_k = 50, top_p = 0.95)
            gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            val_gen_text.extend(gen_text)
        # val_gen_text += [""]*(len(val_processed) - len(val_gen_text))
        val_data["pred"] = val_gen_text
        val_data.to_csv(os.path.join(args.model_path, args.prefix + "val_generated_text.csv"))
        
        acc = []
        for j in range(len(val_data)):
            p = val_data["pred"][j]
            if "Response:" not in p:
                acc.append(0)
                continue
            ct_response = p[p.index("Response:"):]
            ct_response = ct_response.replace("Response: ", "").lstrip().rstrip()
            if "{" in ct_response and "}" in ct_response:
                ind1 = ct_response.index("{")
                ind2 = ct_response.index("}")
            else:
                acc.append(0)
                continue
            # if "{" in ct_response:
            #     ind1 = ct_response.index("{")
            # else:
            #     ct_response = "{" + ct_response
            #     ind1=0
            # if "}" in ct_response:
            #     ind2 = ct_response.index("}")
            # else:
            #     ct_response += "}"
            #     ind2 = ct_response.index("}")
            ct_response1 = ct_response[ind1:ind2+1]
            try:
                ct_dict = eval(ct_response1)
                ct_op = ct_dict["output"]
            except:
                ct_op = "none"

            ct_gold_label = eval(val_data["output"][j])["output"]
            if ct_op == ct_gold_label:
                acc.append(1)
            else:
                acc.append(0)
        print("validation accuracy: ", sum(acc)*100/len(acc))

        
    if args.do_test:
        # running on test
        test_data = pd.read_csv(args.dataset_test)
        test_processed = []
        for i in range(len(test_data)):
            ip1 = prompt_instr + test_data["input"][i]
            ip = f"### Instruction: {ip1}\n ### Response: "
            test_processed.append(ip)

        steps = int(len(test_processed)//batch_size) + 1
        test_gen_text = []
        for i in tqdm(range(steps)):
            if i%100==0:
                print(i, steps)
            try:
                batch = test_processed[i*batch_size:(i+1)*batch_size]
            except:
                batch = test_processed[i*batch_size:]
            if len(batch) == 0:
                continue
            #encoded_input = tokenizer(batch, return_tensors='pt', padding='max_length',
            #                            truncation=True, max_length=512, return_token_type_ids=False).to(device)
            encoded_input = tokenizer(batch, return_tensors='pt', return_token_type_ids=False).to(device)
            gen_tokens = model.generate(**encoded_input, max_new_tokens=1024, do_sample=True, 
                                         top_k = 50, top_p = 0.95)
            gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            test_gen_text.extend(gen_text)
        # test_gen_text += [""]*(len(test_processed) - len(test_gen_text))
        test_data["pred"] = test_gen_text
        test_data.to_csv(os.path.join(args.model_path, args.prefix + "test_generated_text.csv"))
        
        acc = []
        for j in range(len(test_data)):
            p = test_data["pred"][j]
            if "Response:" not in p:
                acc.append(0)
                continue
            ct_response = p[p.index("Response:"):]
            ct_response = ct_response.replace("Response: ", "").lstrip().rstrip()
            if "{" in ct_response and "}" in ct_response:
                ind1 = ct_response.index("{")
                ind2 = ct_response.index("}")
            else:
                acc.append(0)
                continue
            ct_response1 = ct_response[ind1:ind2+1]
            try:
                ct_dict = eval(ct_response1)
                ct_op = ct_dict["output"]
            except:
                ct_op = "none"

            ct_gold_label = eval(test_data["output"][j])["output"]
            if ct_op == ct_gold_label:
                acc.append(1)
            else:
                acc.append(0)
        print("test accuracy: ", sum(acc)*100/len(acc))

if __name__=="__main__":
    main()

'''model = AutoModelForCausalLM.from_pretrained("save_deepspeed/i2ro/gpt2-xl_5_05242024_13_42_13/model")

prompt_instr = "Instruction:\nExplain if Text1 and Text2 have the same author or not, using a dictionary format.\n\n"
ip1 = prompt_instr + a["input"][60] + "\n\nResponse:"
ip2 = prompt_instr + a["input"][61] + "\n\nResponse:"
ip3 = prompt_instr + a["input"][62] + "\n\nResponse:"
ip4 = prompt_instr + a["input"][63] + "\n\nResponse:"
print(ip)
encoded_input = tokenizer([ip1, ip2, ip3, ip4], return_tensors='pt', padding='longest', truncation=True,max_length=1024)
gen_tokens = model.generate(**encoded_input, max_length=1024, do_sample=True, 
                                        top_k = 50, top_p = 0.95)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
gen_text'''
