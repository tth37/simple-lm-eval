from typing import Union, Literal


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
        model = get_llama_lru(model, topk_ratio, recall_ratio, cache_size)
    elif method == 'dense':
        pass
    else:
        raise ValueError(f"Unsupported method: {method}")

    return model.eval(), tokenizer