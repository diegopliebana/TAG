"""Learn Tic-Tac-Toe action values, save them and evaluate a reloaded table."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path

from ..randomness import resolve_seed, seed_argument
import numpy as np
from tag_rust.episodes.collection import collect_episodes
from tag_rust.episodes.gym import GameEnv
from tag_rust.episodes.batch import GameEnvBatch


NATIVE_OPPONENTS = ("first_available", "random", "osla", "osla_python")


class QTable:
    """Estimate learner rewards for each default Tic-Tac-Toe board/action pair.

    Boards use the learner's cells encoding. task['learner']=None trains both
    seats; 0 or 1 selects a fixed seat. The task also records the opponent and
    collection horizon. Encoding identity is checked on every use.
    """
    def __init__(self, task, *, alpha=0.2, gamma=0.99):
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("alpha must be in (0,1] and gamma in [0,1]")
        if (task["game"] != "tictactoe" or task["parameters"] or
                task["learner"] not in (None, 0, 1) or task["opponent_settings"]):
            raise ValueError("table requires default Tic-Tac-Toe, alternating or fixed seats and default opponent settings")
        self.task = deepcopy(task)
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.identity = None
        self.values = {}

    def _check(self, info, *, training=False):
        if self.identity is None:
            self.identity = dict(info["identity"])
        if info["identity"] != self.identity or info["observer"] not in (0, 1):
            raise ValueError("table encoding and observer must match a supported learner decision")
        if training and self.task["learner"] is not None and info["observer"] != self.task["learner"]:
            raise ValueError("training seat must match the recorded task")
        if training and (info["opponent"] != self.task["opponent"] or info["opponent_settings"]):
            raise ValueError("table requires its recorded opponent and default settings")

    @staticmethod
    def _key(observation):
        cells = np.asarray(observation["cells"])
        if cells.shape != (9,) or not np.isin(cells, (-1, 0, 1)).all():
            raise ValueError("table requires default Tic-Tac-Toe cells")
        return tuple(int(cell) for cell in cells)

    @staticmethod
    def _legal(observation):
        mask = np.asarray(observation["action_mask"])
        if mask.shape != (9,) or not np.isin(mask, (0, 1)).all():
            raise ValueError("table requires a nine-entry legal-action mask")
        return np.flatnonzero(mask)

    def choose(self, observation, info, rng, *, epsilon=0.1):
        """Explore with probability epsilon; otherwise sample a best legal action."""
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        self._check(info)
        key, legal = self._key(observation), self._legal(observation)
        if not len(legal):
            raise ValueError("choose requires an ongoing learner decision")
        values = self.values.get(key)
        if rng.random() < epsilon or values is None:
            candidates = legal
        else:
            candidates = legal[values[legal] == values[legal].max()]
        return int(rng.choice(candidates))

    def update(self, transition):
        """Move one entry toward reward plus discounted legal continuation value.

        A rule ending has target=reward. An ongoing or truncated board uses its
        best legal next-action value. gamma applies once per learner step.
        """
        self._check(transition.info, training=True)
        self._check(transition.next_info, training=True)
        if transition.info["observer"] != transition.next_info["observer"]:
            raise ValueError("a transition must retain the same learner viewpoint")
        key = self._key(transition.observation)
        if transition.action not in self._legal(transition.observation):
            raise ValueError("transition action must be legal at its starting decision")
        target = transition.reward
        if not transition.terminated:
            legal = self._legal(transition.next_observation)
            if not len(legal):
                raise ValueError("continuation requires legal next actions")
            following = self.values.get(self._key(transition.next_observation))
            if following is not None:
                target += self.gamma * float(following[legal].max())
        row = self.values.get(key)
        if row is None:
            row = np.zeros(9)
            self.values[key] = row
        row[transition.action] += self.alpha * (target - row[transition.action])

    def save(self, path):
        """Write versioned JSON containing task, encoding, rates and sorted rows."""
        if self.identity is None:
            raise ValueError("use the table with an episode before saving")
        value = dict(format="tag-rust-tictactoe-q", version=2, task=self.task,
                     identity=self.identity, alpha=self.alpha, gamma=self.gamma,
                     rows=[dict(cells=list(key), values=row.tolist())
                           for key, row in sorted(self.values.items())])
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, *, task=None):
        """Read a table's saved task, optionally checking an expected task."""
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value["format"] != "tag-rust-tictactoe-q" or value["version"] not in (1, 2):
            raise ValueError("unsupported table format/version")
        if value["version"] == 1 and value["task"]["learner"] not in (0, 1):
            raise ValueError("version 1 tables require a fixed training seat")
        if task is not None and value["task"] != task:
            raise ValueError("saved table task must match the requested task")
        table = cls(value["task"], alpha=value["alpha"], gamma=value["gamma"])
        if not isinstance(value["identity"], dict) or not value["identity"]:
            raise ValueError("saved table requires an encoding identity")
        table.identity = dict(value["identity"])
        for entry in value["rows"]:
            key = table._key(dict(cells=entry["cells"]))
            row = np.asarray(entry["values"], dtype=np.float64)
            if key in table.values or row.shape != (9,) or not np.isfinite(row).all():
                raise ValueError("saved rows require distinct boards and nine finite values")
            table.values[key] = row.copy()
        return table


def _counts():
    return dict(episodes=0, wins=0, losses=0, draws=0, truncated=0, transitions=0)


def _summary(counts):
    """Report rates per completed episode; an unused seat has None rates."""
    games = counts["episodes"]
    rates = {f"{label}_rate": counts[name] / games if games else None for name, label in
             (("wins", "win"), ("losses", "loss"), ("draws", "draw"), ("truncated", "truncation"))}
    return dict(counts, **rates,
                mean_reward=(counts["wins"] - counts["losses"]) / games if games else None)


def run(table, *, rounds, members, seed, epsilon, learning, opponent=None, learner=None):
    """Collect episodes and report combined and per-seat outcomes.

    Training follows the saved task's seat schedule and opponent. Evaluation
    alternates seats by default; learner=0/1 selects a fixed evaluation seat.
    Logical game number round*members+member determines the alternating seat.
    Each cached batch retains its environments' fixed learner seats and RNGs.
    """
    automatic = seed is None
    seed = resolve_seed(seed)
    if rounds < 1 or not 1 <= members <= 256 or seed < 0:
        raise ValueError("rounds must be positive, members in [1,256], seed nonnegative")
    if learner not in (None, 0, 1):
        raise ValueError("learner must be None, 0 or 1")
    opponent = table.task["opponent"] if opponent is None else opponent
    if learning and opponent != table.task["opponent"]:
        raise ValueError("training opponent must match the recorded task")
    if learning and learner is not None and learner != table.task["learner"]:
        raise ValueError("training seat must match the recorded task")
    selected_seat = table.task["learner"] if learning else learner
    rng = np.random.Generator(np.random.PCG64(seed))
    totals, by_seat = _counts(), {"0": _counts(), "1": _counts()}
    with ExitStack() as stack:
        batches = {}
        def choose(member, observation, info):
            return table.choose(observation, info, rng, epsilon=epsilon)
        for round_index in range(rounds):
            parity = (round_index * members) % 2 if selected_seat is None else 0
            seeds = None
            if parity not in batches:
                environments = []
                for index in range(members):
                    seat = (parity + index) % 2 if selected_seat is None else selected_seat
                    env = GameEnv("tictactoe", learner=seat, opponent=opponent, payloads=("cells",),
                                  max_episode_steps=table.task["max_episode_steps"])
                    stack.callback(env.close)
                    environments.append(env)
                batches[parity] = stack.enter_context(GameEnvBatch(environments))
                seeds = [seed+i for i in range(members)]
            batch = batches[parity]
            for transition in collect_episodes(batch, choose, seeds=seeds):
                if learning:
                    table.update(transition)
                for counts in (totals, by_seat[str(transition.info["observer"])]):
                    counts["transitions"] += 1
                    if transition.terminated or transition.truncated:
                        counts["episodes"] += 1
                        name = ("truncated" if transition.truncated else
                                "wins" if transition.reward > 0 else
                                "losses" if transition.reward < 0 else "draws")
                        counts[name] += 1
    result = dict(_summary(totals), seat_schedule="alternating" if selected_seat is None else f"fixed:{selected_seat}",
                by_seat={seat: _summary(counts) for seat, counts in by_seat.items()})
    if automatic:
        result["seed"] = seed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=128, help="training episodes per member")
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--eval-rounds", type=int, default=16)
    parser.add_argument("--learner", type=int, choices=(0, 1),
                        help="fixed learner seat; default alternates first/second player")
    parser.add_argument("--opponent", choices=NATIVE_OPPONENTS, default="random")
    parser.add_argument("--seed", type=seed_argument, default=None, help="integer or auto; default uses OS randomness")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)
    if args.rounds < 1 or args.eval_rounds < 1 or not 1 <= args.members <= 256 or args.seed < 0:
        parser.error("round counts must be positive, members in [1,256], seed nonnegative")
    if not 0 <= args.epsilon <= 1:
        parser.error("epsilon must be in [0,1]")
    task = dict(game="tictactoe", parameters={}, learner=args.learner,
                opponent=args.opponent, opponent_settings={}, max_episode_steps=args.max_episode_steps)
    table = QTable(task, alpha=args.alpha, gamma=args.gamma)
    baseline = run(table, rounds=args.eval_rounds, members=args.members, seed=args.seed+10000,
                   epsilon=0, learning=False, learner=args.learner)
    training = run(table, rounds=args.rounds, members=args.members, seed=args.seed,
                   epsilon=args.epsilon, learning=True)
    table.save(args.output)
    loaded = QTable.load(args.output, task=task)
    evaluation = run(loaded, rounds=args.eval_rounds, members=args.members, seed=args.seed+10000,
                     epsilon=0, learning=False, learner=args.learner)
    print(json.dumps(dict(seed=args.seed, task=task,
                          settings=dict(rounds=args.rounds, members=args.members,
                                        eval_rounds=args.eval_rounds, epsilon=args.epsilon,
                                        alpha=args.alpha, gamma=args.gamma, eval_seed=args.seed+10000),
                          baseline=baseline, training=training,
                          reloaded_evaluation=evaluation, stored_boards=len(loaded.values),
                          changed_entries=sum(int(np.count_nonzero(row)) for row in loaded.values.values())),
                     indent=2))


if __name__ == "__main__":
    main()
