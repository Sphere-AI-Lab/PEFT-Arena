import argparse
import os
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path
from copy import deepcopy
from utils.extract_answer import extract_answer_boxed, match, extract_overall_answer
import numpy as np
from peft import PeftConfig, PeftModel
import shutil
from transformers import AutoModelForCausalLM

def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_file(input_fp):
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k,v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument('--eval_file', type=str, default='data/m1_eval_data_processed.json')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--tensor_parallel_size', type=int, default=1) 
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)  
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--model_save_name", type=str, required=True)
    parser.add_argument('--output_file_name', type=str, default='raw_results')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args

def is_peft_model(model_name):
    if os.path.exists(os.path.join(model_name, 'adapter_model.safetensors')):
        return True
    return False

def merge_peft_model(model_name):
    peft_config = PeftConfig.from_pretrained(model_name)
    base_model_path = peft_config.base_model_name_or_path
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    peft_model = PeftModel.from_pretrained(base_model, model_name)
    merged_model = peft_model.merge_and_unload()
    # make local tmp dir
    tmp_dir = os.path.join(model_name, 'tmp_model')
    print(f"Saving merged model to {tmp_dir}")
    os.makedirs(tmp_dir, exist_ok=True)
    merged_model.save_pretrained(tmp_dir)
    # also save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(tmp_dir)
    return tmp_dir

def get_model(args):
    if is_peft_model(args.model_name):
        # merge and save as a new tmp model
        tmp_model_path = merge_peft_model(args.model_name)        
        model = LLM(
            model=tmp_model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
            # max_tokens=args.max_tokens,
        )
        tokenizer = AutoTokenizer.from_pretrained(tmp_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, tmp_model_path
    else:
        model = LLM(
            model=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
            # max_tokens=args.max_tokens,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tmp_model_path = None
        return model, tokenizer, tmp_model_path


LLAMA_PROMPT = '<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYor are a helpful assistant.\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

QWEN_PROMPT = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'





def main():

    args = get_args()

    model, tokenizer, tmp_model_path = get_model(args)

    input_data = json.load(open(args.eval_file, 'r')) 

    input_prompt = [item['prompt'] for item in input_data]    
    if args.debug:
        input_prompt = input_prompt[:10]

    if "llama" in args.model_name.lower():
        input_prompt = [LLAMA_PROMPT.format(prompt=item) for item in input_prompt]
    elif "qwen" in args.model_name.lower():
        input_prompt = [QWEN_PROMPT.format(prompt=item) for item in input_prompt]
    else:
        raise NotImplementedError(f"Model {args.model_name} not supported")

    
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )
    outputs = model.generate(input_prompt, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

    output_data = []
    curr_score = 0
    total_count = 0
    for item, output in zip(input_data, outputs):
        item_copy = deepcopy(item)
        item_copy['output'] = output
        item_copy['extracted_output_option'], ans_type = extract_overall_answer(output, item['options'], item_copy['answer_idx'])

        # if ans_type == 1:
        score = item_copy['extracted_output_option'].lower() == item_copy['answer_idx'].lower()
        item_copy['score'] = score
        curr_score += score
        total_count += 1
        output_data.append(item_copy)

    print(f"Total score: {curr_score / total_count}")
    print(f"Total count: {total_count}, total length: {len(input_data)}, total score: {curr_score}")
    if 'checkpoint' in args.model_name and len(args.model_name.split('/')) > 0:
        args.output_file_name = args.output_file_name+'-'+args.model_name.split('/')[-1]
    
    if args.output_dir is not None:
        output_path = Path(args.output_dir) / f"{args.output_file_name}.json"
    else:
        output_path = Path("./results") / "medical" / args.model_save_name / f"{args.output_file_name}.json"
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output_data, open(output_path, 'w'), indent=4)


    if args.debug:
        for output in outputs:
            print(output)
            print("-"*100)
            print("\n")
        exit()

    score_json = {}
    sources = set([item['source'] for item in output_data])
    for source in sources:
        score_json[source] = {
            'acc': sum([item['score'] for item in output_data if item['source'] == source]) / len([item for item in output_data if item['source'] == source]),
            'count': len([item for item in output_data if item['source'] == source]),
            'total_score': sum([item['score'] for item in output_data if item['source'] == source])
        }
    score_json['average'] = np.mean([v['acc'] for v in score_json.values()])
    json.dump(score_json, open(output_path.parent / f"{args.output_file_name}_score.json", 'w'), indent=4)

    print("Score json:")
    for k,v in score_json.items():
        if isinstance(v, dict):
            print(f"{k}: {v['acc']:.4f}")
        else:
            print(f"{k}: {v:.4f}")
    if tmp_model_path is not None:
        print(f"Removing tmp model {tmp_model_path}")
        shutil.rmtree(tmp_model_path)


if __name__ == "__main__":
    main()