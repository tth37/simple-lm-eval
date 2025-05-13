import json
from tqdm import tqdm
from rouge import Rouge
import torch
from typing import Union, Literal
import argparse

def _read_jsonl(file):
    results = []
    with open(file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def _evaluate_rouge(pred, label):
    rouge = Rouge()
    scores = rouge.get_scores(pred, label)[0]
    return {
        "rouge-1": scores['rouge-1']['f'],
        "rouge-2": scores['rouge-2']['f'],
        "rouge-l": scores['rouge-l']['f'],
    }

def _average_score(scores):
    return sum(scores) / len(scores)

def _log_prob_last_token(model, tokenizer, text):
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(model.device)
    context = input_ids[:, :-1]
    target = input_ids[:, -1]
    with torch.no_grad():
        outputs = model(context, use_cache=True)
        past_key_values = outputs.past_key_values
    logits = model(target.unsqueeze(0), past_key_values=past_key_values).logits
    log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
    target_log_prob = log_probs[0, target].item()
    return target_log_prob

def _log_prob(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    return -outputs.loss.item()

def evaluate_xsum(model, tokenizer, limit=0):
    requests = _read_jsonl('data/xsum_1shot.jsonl')
    requests = requests[:limit] if limit > 0 else requests
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(requests, desc="Evaluating", unit="sample")
        
        for request in progress_bar:
            prompt = request['article']
            label = request['summary_gt']
            max_tokens = 64
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
            output = model.generate(
                input_ids=input_ids,
                max_length=len(input_ids[0]) + max_tokens,
                temperature=0.3,
                top_k=0,
                top_p=1,
                do_sample=True
            )
            pred = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
            pred = pred[:pred.find('###')]
            scores = _evaluate_rouge(pred, label)
            
            rouge_1_scores.append(scores['rouge-1'])
            rouge_2_scores.append(scores['rouge-2'])
            rouge_l_scores.append(scores['rouge-l'])
            
            current_rouge_1 = _average_score(rouge_1_scores)
            current_rouge_2 = _average_score(rouge_2_scores)
            current_rouge_l = _average_score(rouge_l_scores)
            
            progress_bar.set_description(
                f"R1: {current_rouge_1:.4f} | R2: {current_rouge_2:.4f} | RL: {current_rouge_l:.4f}"
            )
    
    rouge_1 = _average_score(rouge_1_scores)
    rouge_2 = _average_score(rouge_2_scores)
    rouge_l = _average_score(rouge_l_scores)
    return {
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-l": rouge_l
    }

def evaluate_cnn(model, tokenizer, limit=0):
    requests = _read_jsonl('data/cnn_dailymail_1shot.jsonl')
    requests = requests[:limit] if limit > 0 else requests
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(requests, desc="Evaluating", unit="sample")
        
        for request in progress_bar:
            prompt = request['article']
            label = request['summary_gt']
            max_tokens = 128
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
            output = model.generate(
                input_ids=input_ids,
                max_length=len(input_ids[0]) + max_tokens,
                temperature=0.3,
                top_k=0,
                top_p=1,
                do_sample=True
            )
            pred = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
            pred = pred[:pred.find('###')]
            scores = _evaluate_rouge(pred, label)
            
            rouge_1_scores.append(scores['rouge-1'])
            rouge_2_scores.append(scores['rouge-2'])
            rouge_l_scores.append(scores['rouge-l'])
            
            current_rouge_1 = _average_score(rouge_1_scores)
            current_rouge_2 = _average_score(rouge_2_scores)
            current_rouge_l = _average_score(rouge_l_scores)
            
            progress_bar.set_description(
                f"R1: {current_rouge_1:.4f} | R2: {current_rouge_2:.4f} | RL: {current_rouge_l:.4f}"
            )
    
    rouge_1 = _average_score(rouge_1_scores)
    rouge_2 = _average_score(rouge_2_scores)
    rouge_l = _average_score(rouge_l_scores)
    return {
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-l": rouge_l
    }

def evaluate_piqa(model, tokenizer, limit=0):
    requests = _read_jsonl('data/piqa_0shot.jsonl')
    requests = requests[:limit] if limit > 0 else requests
    correct = 0
    progress_bar = tqdm(requests, desc="Evaluating", unit="sample")
    for request in progress_bar:
        text0 = f"Question: {request['goal']}\nAnswer: {request['sol1']}"
        text1 = f"Question: {request['goal']}\nAnswer: {request['sol2']}"
        prob0 = _log_prob_last_token(model, tokenizer, text0)
        prob1 = _log_prob_last_token(model, tokenizer, text1)
        predicted_label = 0 if prob0 > prob1 else 1
        if predicted_label == request['label']:
            correct += 1
        progress_bar.set_description(f"Accuracy: {correct / (progress_bar.n + 1):.4f}")
    return {
        "accuracy": correct / len(requests)
    }

def evaluate_boolq(model, tokenizer, limit=0):
    requests = _read_jsonl('data/boolq_0shot.jsonl')
    requests = requests[:limit] if limit > 0 else requests
    correct = 0
    progress_bar = tqdm(requests, desc="Evaluating", unit="sample")
    for request in progress_bar:
        text0 = f"{request['passage']}\nQuestion: {request['question']}\nAnswer: no"
        text1 = f"{request['passage']}\nQuestion: {request['question']}\nAnswer: yes"
        prob0 = _log_prob_last_token(model, tokenizer, text0)
        prob1 = _log_prob_last_token(model, tokenizer, text1)
        predicted_label = 0 if prob0 > prob1 else 1
        if predicted_label == request['answer']:
            correct += 1
        progress_bar.set_description(f"Accuracy: {correct / (progress_bar.n + 1):.4f}")
    return {
        "accuracy": correct / len(requests)
    }

def evaluate(
    model,
    tokenizer,
    task,
    limit=0
):
    if task == 'xsum':
        return evaluate_xsum(model, tokenizer, limit)
    elif task == 'cnn':
        return evaluate_cnn(model, tokenizer, limit)
    elif task == 'piqa':
        return evaluate_piqa(model, tokenizer, limit)
    elif task == 'boolq':
        return evaluate_boolq(model, tokenizer, limit)
    else:
        raise NotImplementedError(f"Task {task} not implemented.")

def load_model(
    backend: Union[Literal['transformers'], Literal['modelscope']],
    model_name: str,
    method: Union[Literal['griffin'], Literal['lru']],
    topk_ratio: float,
    recall_ratio: float,
    cache_size: int,
    mode: Union[Literal['gen'], Literal['class']]
):
    if backend == 'transformers':
        from transformers import AutoModelForCausalLM, AutoTokenizer
    elif backend == 'modelscope':
        from modelscope import AutoModelForCausalLM, AutoTokenizer
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float16",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if method == 'griffin':
        from griffin import get_llama_griffin
        model = get_llama_griffin(model, topk_ratio, mode)
    elif method == 'lru':
        from lru import get_llama_lru
        model = get_llama_lru(model, topk_ratio, recall_ratio, cache_size)
    elif method == 'dense':
        pass
    else:
        raise ValueError(f"Unsupported method: {method}")

    return model.eval(), tokenizer

args = argparse.ArgumentParser()
args.add_argument('--backend', type=str, default='transformers', choices=['transformers', 'modelscope'], help='Backend to use for loading the model.')
args.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B', help='Model name to load.')
args.add_argument('--method', type=str, default='griffin', choices=['griffin', 'lru', 'dense'], help='Method to use for model reduction.')
args.add_argument('--topk_ratio', type=float, default=0.5, help='Top-k ratio for model reduction.')
args.add_argument('--recall_ratio', type=float, default=0.5, help='Recall ratio for model reduction.')
args.add_argument('--cache_size', type=int, default=4, help='Cache size for model reduction.')
args.add_argument('--task', type=str, default='xsum', choices=['xsum', 'cnn', 'piqa', 'boolq'], help='Task to evaluate.')
args.add_argument('--limit', type=int, default=0, help='Limit for the number of samples to evaluate.')
args = args.parse_args()

args.mode = 'class' if args.task in ['piqa'] else 'gen'

model, tokenizer = load_model(
    backend=args.backend,
    model_name=args.model_name,
    method=args.method,
    topk_ratio=args.topk_ratio,
    recall_ratio=args.recall_ratio,
    cache_size=args.cache_size,
    mode=args.mode
)
result = evaluate(
    model=model,
    tokenizer=tokenizer,
    task=args.task,
    limit=args.limit
)
print(f"model_name: {args.model_name}")
print(f"method: {args.method}")
print(f"topk_ratio: {args.topk_ratio}")
print(f"recall_ratio: {args.recall_ratio}")
print(f"cache_size: {args.cache_size}")
print(f"task: {args.task}")
print(f"limit: {args.limit}")
print(f"Evaluation result for task {args.task}:")
for key, value in result.items():
    print(f"{key}: {value:.4f}")
