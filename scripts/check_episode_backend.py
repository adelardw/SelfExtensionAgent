"""Tiny offline MLX gradient check; no model downloads or benchmark claims.

Run with .venv-rl/bin/python scripts/check_episode_backend.py on Apple Silicon.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm.tuner.lora import LoRALinear
    from mlxrl.algo import GRPOAlgorithm
    from mlxrl.policy.logprobs import completion_logprobs
    from mlxrl.train import optimizer_step
    from src.agent_train.episode_grpo import build_batch

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 8)
            self.output = nn.Linear(8, 32, bias=False)
            self.freeze()
            self.output = LoRALinear.from_base(self.output, r=2, scale=2)
        def __call__(self, tokens): return self.output(self.embedding(tokens))

    mx.random.seed(7)
    model = TinyPolicy()
    model.eval()
    def step(prompt, completion):
        captured = completion_logprobs(model, [prompt], [completion]).logprobs[0].tolist()
        return {'sample': {'prompt_tokens': prompt, 'completion_tokens': completion,
                           'old_policy_logprobs': captured}}
    episodes = [
        {'task_id': 'same', 'split': 'train', 'training_eligible': True, 'reward': 1.,
         'steps': [step([1, 2, 3], [4, 5]), step([1, 2, 3, 4, 5, 20, 21], [6])]},
        {'task_id': 'same', 'split': 'train', 'training_eligible': True, 'reward': 0.,
         'steps': [step([1, 2, 3], [7, 8, 9])]},
    ]
    batch = build_batch(model, episodes, 0, .02, False)
    assert batch.mask.tolist() == [[1., 1., 0.], [1., 0., 0.], [1., 1., 1.]]
    assert abs(batch.advantages[0].item()-batch.advantages[1].item()) < 1e-6
    assert batch.advantages[2].item() < 0
    assert batch.prompt_token_ids[1] == (1, 2, 3, 4, 5, 20, 21)
    # Observation tokens 20,21 remain context and never become loss targets.
    assert all(20 not in row and 21 not in row for row in batch.completion_token_ids)
    initial = model.output.lora_b.tolist()
    base = model.embedding.weight.tolist()
    result = optimizer_step(model, optim.Adam(learning_rate=1e-3), batch, beta=.02,
        pad_token_id=0, algorithm=GRPOAlgorithm(), micro_batch_size=1)
    assert mx.isfinite(mx.array(result.loss)).item()
    assert model.output.lora_b.tolist() != initial
    assert model.embedding.weight.tolist() == base
    print(json.dumps({'check': 'passed', 'action_tokens': int(batch.mask.sum().item()),
        'observation_tokens_trained': 0, 'lora_changed': True, 'base_unchanged': True,
        'metrics': asdict(result)}))


if __name__ == '__main__': main()
