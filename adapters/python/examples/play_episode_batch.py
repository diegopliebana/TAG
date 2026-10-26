"""Collect complete independent episodes using explicit masks and ending flags."""
import argparse
from tag_rust.randomness import resolve_seed, seed_argument
import numpy as np
from tag_rust.episodes.gym import GameEnv
from tag_rust.episodes.batch import GameEnvBatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=seed_argument, default=None)
    seed = resolve_seed(parser.parse_args().seed)
    print(f"Episode master seed: {seed}")
    environments = [GameEnv("connect4", learner=i % 2, opponent=policy, payloads=("cells",))
                    for i, policy in enumerate(("random", "osla_python", "osla", "first_available"))]
    try:
        with GameEnvBatch(environments) as batch:
            result = batch.reset(seeds=[(seed+i) % 2**64 for i in range(len(environments))])
            if any(result.errors):
                raise RuntimeError(result.errors)
            active = result.valid.copy()
            rewards = np.zeros(len(environments))
            while active.any():
                actions = [int(np.flatnonzero(result.observations["action_mask"][i])[0])
                           if active[i] else None for i in range(len(environments))]
                result = batch.step(actions)
                if any(result.errors):
                    raise RuntimeError(result.errors)
                finished = result.valid & (result.terminated | result.truncated)
                rewards[finished] = result.rewards[finished]
                active[finished] = False
            print("learner rewards:", rewards.tolist())
    finally:
        for environment in environments:
            environment.close()


if __name__ == "__main__":
    main()
