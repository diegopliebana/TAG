"""Play one Sushi Go! learner episode or a batch with rotating learner seats."""
import argparse
import numpy as np
from tag_rust.randomness import resolve_seed, seed_argument
from tag_rust.episodes.gym import GameEnv
from tag_rust.episodes.batch import GameEnvBatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=seed_argument, default=None)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    seed = resolve_seed(args.seed)
    seats = range(3) if args.batch else (0,)
    environments = [GameEnv("sushigo", players=3, learner=seat, opponent="random")
                    for seat in seats]
    print(f"master seed={seed}")
    try:
        if args.batch:
            with GameEnvBatch(environments) as batch:
                result = batch.reset(seeds=[(seed + i) % 2**64 for i in seats])
                if any(result.errors):
                    raise RuntimeError(result.errors)
                active = result.valid.copy()
                rewards = np.zeros(len(environments))
                while active.any():
                    actions = [int(np.flatnonzero(result.observations["action_mask"][i])[0])
                               if active[i] else None for i in seats]
                    result = batch.step(actions)
                    if any(result.errors):
                        raise RuntimeError(result.errors)
                    finished = result.valid & (result.terminated | result.truncated)
                    rewards[finished] = result.rewards[finished]
                    active[finished] = False
                print("learner rewards:", rewards.tolist())
        else:
            environment = environments[0]
            observation, info = environment.reset(seed=seed)
            while True:
                action = int(np.flatnonzero(observation["action_mask"])[0])
                observation, reward, terminated, truncated, info = environment.step(action)
                if terminated or truncated:
                    print("learner reward:", reward, "results:", info["results"].tolist(),
                          "steps:", info["learner_steps"])
                    break
    finally:
        for environment in environments:
            environment.close()


if __name__ == "__main__":
    main()
