from torch import nn
import torch

class LRUReducedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, topk_ratio, recall_ratio, cache_size, device, dtype, mode):
        super(LRUReducedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.topk_ratio = float(topk_ratio)
        self.topk_size = int(self.in_features * self.topk_ratio)
        self.recall_ratio = float(recall_ratio)
        self.recall_size = int(self.topk_size * self.recall_ratio)
        self.cache_size = int(cache_size)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('cached_weight', torch.zeros(cache_size, out_features, self.topk_size, device=device, dtype=dtype))
        self.register_buffer('cached_mask', torch.zeros(cache_size, in_features, dtype=torch.bool, device=device))
        self.register_buffer('cached_indices', torch.zeros(cache_size, self.topk_size, dtype=torch.long, device=device))
        self.register_buffer('lru', torch.zeros(cache_size, dtype=torch.long, device=device))
        self.lru_counter = 0
        self.lru_valid = 0
        self.mode = mode
        assert self.mode in ['gen', 'class'], f"Unsupported mode: {self.mode}"

    @classmethod
    def from_linear(cls, linear, topk_ratio, recall_ratio, cache_size, mode):
        in_features = linear.in_features
        out_features = linear.out_features
        bias = linear.bias is not None
        device = linear.weight.device
        dtype = linear.weight.dtype
        lru_reduced_linear = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            topk_ratio=topk_ratio,
            recall_ratio=recall_ratio,
            cache_size=cache_size,
            device=device,
            dtype=dtype,
            mode=mode
        )
        lru_reduced_linear.weight.data = linear.weight.data
        if bias:
            lru_reduced_linear.bias.data = linear.bias.data
        return lru_reduced_linear

    @torch.no_grad()
    def _get_slice(self, indices):
        # Get the slice of the weight matrix corresponding to the top-k indices
        return self.weight[:, indices]

    @torch.no_grad()
    def _cache_slice(self, indices, mask):
        if self.lru_valid < self.cache_size:
            lru_idx = self.lru_valid
            self.lru_valid += 1
        else:
            lru_idx = torch.argmin(self.lru)
        
        self.cached_weight[lru_idx] = self._get_slice(indices)
        self.cached_mask[lru_idx] = mask
        self.cached_indices[lru_idx] = indices

        return lru_idx

    @torch.no_grad()
    def forward(self, x):
        if self.mode == 'gen':
            bsz, seq, _ = x.shape
            if seq != 1:
                x = x.view(bsz * seq, -1)
                result = x @ self.weight.T
                if self.bias is not None:
                    result += self.bias
                result = result.view(bsz, seq, -1)
                return result
            
            x = x.view(bsz, -1)

            self.lru_counter += 1

            # Get top-k indices based on sum of absolute values along columns
            scores = torch.sum(torch.abs(x), dim=0)
            _, indices = torch.topk(scores, k=self.topk_size, dim=0, sorted=False)

            mask = torch.zeros_like(scores, dtype=torch.bool, device=x.device)
            mask.scatter_(0, indices, True)

            max_recall, max_idx = (mask & self.cached_mask).sum(dim=1).max(dim=0)

            if max_recall < self.recall_size:
                max_idx = self._cache_slice(indices, mask)

            self.lru[max_idx] = self.lru_counter

            cached_weight = self.cached_weight[max_idx]
            cached_indices = self.cached_indices[max_idx]

            result = x[:, cached_indices] @ cached_weight.T
            if self.bias is not None:
                result += self.bias

            result = result.view(bsz, 1, -1)
            return result
        elif self.mode == 'class':
            
            bsz, seq, _ = x.shape
            assert seq > 1
            assert bsz == 1
            x = x.view(seq, -1)
            prompt_scores = torch.sum(torch.abs(x[:-1]), dim=0)
            _, prompt_indices = torch.topk(prompt_scores, k=self.topk_size, dim=0, sorted=False)
            prompt_mask = torch.zeros_like(prompt_scores, dtype=torch.bool, device=x.device)
            prompt_mask.scatter_(0, prompt_indices, True)

            target_scores = torch.sum(torch.abs(x[-1].unsqueeze(0)), dim=0)
            _, target_indices = torch.topk(target_scores, k=self.topk_size, dim=0, sorted=False)
            target_mask = torch.zeros_like(target_scores, dtype=torch.bool, device=x.device)
            target_mask.scatter_(0, target_indices, True)

            recall = (prompt_mask & target_mask).sum(dim=0)
            if recall < self.recall_size:
                mask = target_mask
            else:
                mask = prompt_mask

            x[-1] = x[-1].masked_fill(mask, 0.0)
            result = x @ self.weight.T
            if self.bias is not None:
                result += self.bias
            
            result = result.view(bsz, seq, -1)
            return result
    
def get_llama_lru(model, topk_ratio, recall_ratio, cache_size, mode):
    """
    Get LRU reduced linear layer from a model.
    """
    for layer in model.model.layers:
        layer.mlp.gate_proj = LRUReducedLinear.from_linear(
            layer.mlp.gate_proj,
            topk_ratio=topk_ratio,
            recall_ratio=recall_ratio,
            cache_size=cache_size,
            mode=mode
        )
        layer.mlp.up_proj = LRUReducedLinear.from_linear(
            layer.mlp.up_proj,
            topk_ratio=topk_ratio,
            recall_ratio=recall_ratio,
            cache_size=cache_size,
            mode=mode
        )
        layer.mlp.down_proj = LRUReducedLinear.from_linear(
            layer.mlp.down_proj,
            topk_ratio=topk_ratio,
            recall_ratio=recall_ratio,
            cache_size=cache_size,
            mode=mode
        )
    return model