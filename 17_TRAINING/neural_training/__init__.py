"""PyTorch optimization step adapter; imported only when training is requested."""
from __future__ import annotations

def train_causal_batch(model, optimizer, input_ids, *, grad_clip: float = 1.0, loss_mask=None):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("install the neural extra to run PyTorch training") from exc
    if input_ids.ndim != 2 or input_ids.shape[1] < 2: raise ValueError("input_ids must be [B,T] with T>=2")
    if grad_clip <= 0: raise ValueError("grad_clip must be positive")
    model.train(); optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    if isinstance(logits, dict): logits = logits["logits"]
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("model must return logits shaped [B,T,V]")
    flat_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                input_ids[:, 1:].reshape(-1), reduction="none")
    if loss_mask is not None:
        if loss_mask.shape != input_ids.shape: raise ValueError("loss_mask shape must match input_ids")
        mask = loss_mask[:, 1:].reshape(-1).to(flat_loss.dtype)
        loss = (flat_loss * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss = flat_loss.mean()
    if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(torch.as_tensor(grad_norm)): raise FloatingPointError("non-finite gradient norm")
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm)}

__all__ = ["train_causal_batch"]
