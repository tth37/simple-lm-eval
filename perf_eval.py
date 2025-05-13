from typing import Union, Literal
import torch


def load_model(
    backend: Union[Literal['transformers'], Literal['modelscope']],
    model_name: str,
    method: Union[Literal['griffin'], Literal['lru']],
    topk_ratio: float,
    recall_ratio: float,
    cache_size: int,
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
        model = get_llama_griffin(model, topk_ratio, 'gen')
    elif method == 'lru':
        from lru import get_llama_lru
        model = get_llama_lru(model, topk_ratio, recall_ratio, cache_size, 'gen')
    elif method == 'dense':
        pass
    else:
        raise ValueError(f"Unsupported method: {method}")

    return model.eval(), tokenizer

def benchmark(fn, warmup_iters=10, benchmark_iters=100):
    """Benchmark a function using CUDA events."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(warmup_iters):
        _ = fn()
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_event.record()
    for _ in range(benchmark_iters):
        _ = fn()
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / benchmark_iters
    return elapsed_time

batch_size = 2
model, tokenizer = load_model(
    backend='modelscope',
    model_name='Qwen/Qwen3-1.7B',
    method='griffin',
    topk_ratio=0.5,
    recall_ratio=0.01,
    cache_size=1
)
input_ids = torch.randint(0, 100, (batch_size, 1)).to(model.device)
def forward_fn():
    with torch.no_grad():
        output = model(input_ids)

latency = benchmark(forward_fn)
print(f"Latency: {latency:.4f} ms")
