# scripts/eval_surprisal.py
import argparse, math, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def logprobs_for_continuation(model, tok, context, continuation, device="cpu"):
    # Build input = [context + continuation], get log P(continuation | context)
    ctx_ids = tok(context, return_tensors="pt").to(device)
    cont_ids = tok(continuation, return_tensors="pt", add_special_tokens=False).to(device)
    # Concatenate for a single forward pass
    input_ids = torch.cat([ctx_ids["input_ids"], cont_ids["input_ids"]], dim=1)
    attn_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    # Shift for next-token prediction
    logits = out.logits[:, :-1, :]
    tgt_ids = input_ids[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)

    # We only score tokens that belong to the continuation
    # Those positions are the last len(cont_ids) tokens of tgt_ids
    T = cont_ids["input_ids"].shape[1]
    logps = []
    for pos in range(input_ids.shape[1] - T, input_ids.shape[1]):
        token_id = tgt_ids[0, pos]
        logp = logprobs[0, pos, token_id].item()
        logps.append(logp)
    return logps  # list of per-token log p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g., gpt2, gpt2-xl, facebook/opt-1.3b")
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(args.device).eval()

    ds = load_dataset("tonyhong/CSK")[args.split]

    def avg_surprisal(logps):  # negative avg log prob in nats
        return -sum(logps) / max(1, len(logps))

    rows = []
    for ex in ds:
        ctx = ex["before B"].strip()
        b = ex["B"].strip()
        logs = logprobs_for_continuation(model, tok, ctx, b, device=args.device)
        rows.append({
            "item": ex["item"],
            "condition": ex["condition type"],
            "tokens": len(logs),
            "avg_surprisal": avg_surprisal(logs),
        })

    # Aggregate by condition
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.groupby("condition")["avg_surprisal"].mean().sort_values())
    # Optional: save CSV
    df.to_csv(f"results_{args.model.replace('/','_')}.csv", index=False)

if __name__ == "__main__":
    main()
