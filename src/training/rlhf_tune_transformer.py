import argparse
import glob
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    REWARD_EMBED_DIM,
    REWARD_HIDDEN_DIM,
    REWARD_MODEL_CHECKPOINT,
    RLHF_BATCH_SIZE,
    RLHF_ITERATIONS,
    RLHF_KL_COEF,
    RLHF_LEARNING_RATE,
    RLHF_MAX_NEW_TOKENS,
    RLHF_PRIMER_LENGTH,
    SPLIT_DIR,
    TOP_K,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_MAX_SEQ_LEN,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    VOCAB_PATH,
)
from src.generation.generate_transformer import get_valid_tokens_and_ids, load_vocab
from src.models.reward_model import MusicRewardModel
from src.models.transformer import MusicTransformer


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PrimerDataset(Dataset):
    def __init__(self, json_path: Path, primer_length: int):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.primers = []
        for item in data:
            ids = item["token_ids"]
            if len(ids) >= primer_length:
                self.primers.append(ids[:primer_length])

        if not self.primers:
            raise ValueError("No valid primers found.")

    def __len__(self):
        return len(self.primers)

    def __getitem__(self, idx):
        return torch.tensor(self.primers[idx], dtype=torch.long)


def sample_next_with_logprob(
    model,
    generated,
    valid_ids,
    temperature,
    top_k,
):
    context = generated[:, -model.max_seq_len :] if generated.size(1) > model.max_seq_len else generated

    logits = model(context)[:, -1, :]
    logits = logits / temperature

    mask = torch.full_like(logits, float("-inf"))
    mask[:, valid_ids] = 0.0
    logits = logits + mask

    if top_k is not None and top_k > 0:
        top_values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        kth = top_values[:, -1].unsqueeze(1)
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)

    next_token = dist.sample().unsqueeze(1)
    logprob = dist.log_prob(next_token.squeeze(1))

    return next_token, logprob


def logprob_of_token(model, generated, token, valid_ids, temperature):
    context = generated[:, -model.max_seq_len :] if generated.size(1) > model.max_seq_len else generated

    logits = model(context)[:, -1, :]
    logits = logits / temperature

    mask = torch.full_like(logits, float("-inf"))
    mask[:, valid_ids] = 0.0
    logits = logits + mask

    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(1, token).squeeze(1)


def rollout_policy(
    policy,
    ref_policy,
    primers,
    vocab,
    max_new_tokens,
    temperature,
    top_k,
):
    valid_time_shift = get_valid_tokens_and_ids(vocab, "TIME_SHIFT")
    valid_note_on = get_valid_tokens_and_ids(vocab, "NOTE_ON")
    valid_duration = get_valid_tokens_and_ids(vocab, "DURATION")
    valid_velocity = get_valid_tokens_and_ids(vocab, "VELOCITY")

    pattern = [
        [idx for _, idx in valid_time_shift],
        [idx for _, idx in valid_note_on],
        [idx for _, idx in valid_duration],
        [idx for _, idx in valid_velocity],
    ]

    generated = primers
    policy_logprobs = []
    ref_logprobs = []

    for _ in range(max_new_tokens):
        valid_ids = pattern[generated.size(1) % 4]

        next_token, logprob = sample_next_with_logprob(
            model=policy,
            generated=generated,
            valid_ids=valid_ids,
            temperature=temperature,
            top_k=top_k,
        )

        with torch.no_grad():
            ref_lp = logprob_of_token(
                model=ref_policy,
                generated=generated,
                token=next_token,
                valid_ids=valid_ids,
                temperature=temperature,
            )

        policy_logprobs.append(logprob)
        ref_logprobs.append(ref_lp)

        generated = torch.cat([generated, next_token], dim=1)

    policy_logprobs = torch.stack(policy_logprobs, dim=1).sum(dim=1)
    ref_logprobs = torch.stack(ref_logprobs, dim=1).sum(dim=1)

    return generated, policy_logprobs, ref_logprobs


def parse_args():
    parser = argparse.ArgumentParser(description="RLHF fine-tune Transformer with reward model.")
    parser.add_argument("--policy-checkpoint", type=str, required=True)
    parser.add_argument("--reward-checkpoint", type=str, default=REWARD_MODEL_CHECKPOINT.name)
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-checkpoint", type=str, default="transformer_rlhf_best.pt")
    parser.add_argument("--max-new-tokens", type=int, default=RLHF_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    vocab, _ = load_vocab(PROCESSED_DIR / VOCAB_PATH.name)
    vocab_size = len(vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    policy = MusicTransformer(
        vocab_size=vocab_size,
        max_seq_len=TRANSFORMER_MAX_SEQ_LEN,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_FF_DIM,
        dropout=TRANSFORMER_DROPOUT,
    ).to(device)

    ref_policy = MusicTransformer(
        vocab_size=vocab_size,
        max_seq_len=TRANSFORMER_MAX_SEQ_LEN,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_FF_DIM,
        dropout=TRANSFORMER_DROPOUT,
    ).to(device)

    policy_ckpt = torch.load(CHECKPOINT_DIR / args.policy_checkpoint, map_location=device)
    policy_state = policy_ckpt["model_state_dict"] if "model_state_dict" in policy_ckpt else policy_ckpt

    policy.load_state_dict(policy_state)
    ref_policy.load_state_dict(policy_state)
    ref_policy.eval()

    for p in ref_policy.parameters():
        p.requires_grad = False

    reward_model = MusicRewardModel(
        vocab_size=vocab_size,
        embed_dim=REWARD_EMBED_DIM,
        hidden_dim=REWARD_HIDDEN_DIM,
    ).to(device)

    reward_ckpt = torch.load(CHECKPOINT_DIR / args.reward_checkpoint, map_location=device)
    reward_model.load_state_dict(reward_ckpt["model_state_dict"])
    reward_model.eval()

    for p in reward_model.parameters():
        p.requires_grad = False

    dataset = PrimerDataset(SPLIT_DIR / args.input_json, RLHF_PRIMER_LENGTH)
    loader = DataLoader(dataset, batch_size=RLHF_BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=RLHF_LEARNING_RATE)

    moving_baseline = 0.0
    best_reward = -1.0

    step = 0
    while step < RLHF_ITERATIONS:
        for primers in loader:
            if step >= RLHF_ITERATIONS:
                break

            primers = primers.to(device)

            generated, policy_lp, ref_lp = rollout_policy(
                policy=policy,
                ref_policy=ref_policy,
                primers=primers,
                vocab=vocab,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            with torch.no_grad():
                rewards = reward_model(generated)
                avg_reward = rewards.mean().item()

            moving_baseline = 0.9 * moving_baseline + 0.1 * avg_reward
            advantages = rewards - moving_baseline

            policy_loss = -(advantages.detach() * policy_lp).mean()
            kl_loss = (policy_lp - ref_lp).mean()

            loss = policy_loss + RLHF_KL_COEF * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            step += 1

            print(
                f"Step {step:04d} | "
                f"Reward: {avg_reward:.4f} | "
                f"Policy Loss: {policy_loss.item():.4f} | "
                f"KL: {kl_loss.item():.4f} | "
                f"Total: {loss.item():.4f}"
            )

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(
                    {
                        "model_state_dict": policy.state_dict(),
                        "best_reward": best_reward,
                        "vocab_size": vocab_size,
                    },
                    CHECKPOINT_DIR / args.output_checkpoint,
                )
                print(f"Saved RLHF checkpoint: {CHECKPOINT_DIR / args.output_checkpoint}")


if __name__ == "__main__":
    main()