import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

from examples import get_examples


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: \nLet's think step by step.\n", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    # "abel": (
    #     "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
    #     "{output}",
    #     "\n\n",
    # ),
    "abel": (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. The time limit is set to 20,480 tokens. If the assistant's response exceeds this limit, a progressively increasing penalty with the number of tokens exceeded will be applied.\nuser\nSolve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n{input}\nRemember to put your answer on its own line after \"Answer:\".\nassistant",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "llama-boxed": (
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "{output}",
        "\n\n",
    ),
    "qwen-meta":(
        "<|im_start|>system\nYou are a helpful assistant. Before solving any problem, you must start by reflecting on your metacognitive strategy in the <think> section. In <think>, do not solve the problem yet. Instead, analyze what type of problem it is and what concept it tests, what key information and constraints are given, what general class of problems it belongs to and why, what reasoning method is most appropriate (e.g., algebraic, combinatorial, case analysis), and what common pitfalls to avoid. Finally, summarize a general, reusable approach or solution template that can be applied to this type of problem. After that, in the <answer> section, solve the problem step-by-step, clearly showing all work, and conclude with your final answer in \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}\nYour output must strictly follow this format: \n<think> high-level metacognitive reasoning, problem type classification, and generalized solving strategy here </think>\n<answer> detailed step-by-step solution here, ending with \\boxed{{}} </answer> <|im_end|>\n"
        "<|im_start|>assistant\n<think>",
        "{output}",
        "\n\n",
""
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "{output}",
        "\n\n",
    ),
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
    "o1_cot": (
        '[Round 0] USER:\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}. ASSISTANT:\n',
        "{output}",
        "\n\n"
    ),
    "deepseek-r1": (
        'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {input}. Assistant:',
        "{output}",
        "\n\n"
    ),
    "openr1-qwen": (
        "<|im_start|>system\nYour task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {{thoughts}} </think>\n\". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After \"</think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{{}} for closed-form results like multiple choices or mathematical solutions.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen-think": (
        "<|im_start|>system\nYour task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: “<think>\n thoughts </think>\n”. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After “</think>\n” in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input} Let's think step by step and output the final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "openr1-llama": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 11 Nov 2025<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\nLet's think step by step.",
        "{output}",
        "\n\n",
    ),
}


def construct_prompt(example, data_name, args):
    if args.adapt_few_shot and data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        demos = load_prompt(data_name, args.prompt_type, 5)
    else:
        demos = load_prompt(data_name, args.prompt_type, args.num_shots)
    prompt_type = args.prompt_type
    if prompt_type == "platypus_fs":
        prompt_type = "cot"
    if prompt_type == "tool-integrated":
        prompt_type = "tora"

    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    if args.prompt_type == "qwen25-math-cot":
        # Hotfix to support putting all demos into a single turn
        demo_prompt = splitter.join([q + "\n" + a for q, a in demos])
    else:
        demo_prompt = splitter.join(
            [
                input_template.format(input=q) + output_template.format(output=a)
                for q, a in demos
            ]
        )
    context = input_template.format(input=example["question"])
    if len(demo_prompt) == 0 or (
        args.adapt_few_shot and example["gt_ans"] not in ["A", "B", "C", "D", "E"]
    ):
        full_prompt = context
    else:
        if args.prompt_type == "qwen25-math-cot":
            # Hotfix to supportting put all demos into a single turn
            full_prompt = demo_prompt + splitter + example["question"]
            full_prompt = input_template.format(input=full_prompt)
        else:
            full_prompt = demo_prompt + splitter + context

    if args.prompt_type == "platypus_fs":
        full_prompt_temp = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt_temp.format(instruction=full_prompt)

    if prompt_type == "tora":
        full_prompt = (
            """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
            + full_prompt
        )
    print(full_prompt[0])
    return full_prompt.strip(" ")  # important!


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
