"""Run one masked single-learner episode; see the Gym episode guide for semantics."""
import argparse
from tag_rust.randomness import resolve_seed, seed_argument
from tag_rust.episodes.gym import GameEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=seed_argument, default=None)
    seed = resolve_seed(parser.parse_args().seed)
    print(f"Episode master seed: {seed}")
    env = GameEnv("connect4", learner=1, opponent="random", payloads=("board",))
    try:
        observation, info = env.reset(seed=seed)
        env.action_space.seed((seed+1) % 2**64)
        while True:
            action = env.action_space.sample(mask=observation["action_mask"])
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"learner reward={reward}, results={info['results'].tolist()}, steps={info['learner_steps']}")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
