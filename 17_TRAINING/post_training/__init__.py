"""Four-head multi-token prediction objective from ADA v4.5."""
from __future__ import annotations

def multi_token_prediction_loss(main_logits, mtp_logits, input_ids, *, coefficient: float = 0.3):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for MTP loss") from exc
    if input_ids.ndim != 2 or main_logits.ndim != 3: raise ValueError("invalid LM tensor shapes")
    if len(mtp_logits) != 4: raise ValueError("ADA v4.5 expects exactly four MTP heads")
    if not 0 <= coefficient <= 1: raise ValueError("MTP coefficient must be in [0,1]")
    main = F.cross_entropy(main_logits[:, :-1].reshape(-1, main_logits.size(-1)),
                           input_ids[:, 1:].reshape(-1))
    losses = []
    length = input_ids.size(1)
    for index, logits in enumerate(mtp_logits):
        offset = index + 2
        if logits.shape[:2] != main_logits.shape[:2] or length <= offset:
            raise ValueError("MTP logits or sequence length invalid")
        losses.append(F.cross_entropy(
            logits[:, :length-offset].reshape(-1, logits.size(-1)),
            input_ids[:, offset:].reshape(-1),
        ))
    mtp = torch.stack(losses).mean()
    return main + coefficient * mtp, {"main": main.detach(), "mtp": mtp.detach(), "heads": tuple(x.detach() for x in losses)}

__all__ = ["multi_token_prediction_loss"]
