"""Collection parity, owned records, failure boundaries and Q-learning arithmetic."""
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from tag_rust.collection import CollectionError, Transition, collect_episodes
from tag_rust.gym import GameEnv
from tag_rust.gym_batch import GameEnvBatch


SPEC = importlib.util.spec_from_file_location(
    "tabular_q_example", Path(__file__).parents[1] / "examples/tabular_q.py")
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)
QTable = EXAMPLE.QTable


def first(member, observation, info):
    return int(np.flatnonzero(observation["action_mask"])[0])


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, Transition):
        return {key: plain(item) for key, item in vars(value).items()}
    return value


class CollectionTests(unittest.TestCase):
    def test_complete_traces_match_individual_games_seats_and_seed_continuation(self):
        for game in ("tictactoe", "connect4"):
            with self.subTest(game=game), ExitStack() as stack:
                members, singles = [], []
                for i, opponent in enumerate(("first_available", "random", "osla", "osla_python")):
                    config = dict(learner=i % 2, opponent=opponent, payloads=("cells", "board"),
                                  max_episode_steps=1 if i >= 2 else None)
                    for group in (members, singles):
                        env = GameEnv(game, **config)
                        stack.callback(env.close)
                        group.append(env)
                batch = stack.enter_context(GameEnvBatch(members))
                for seeds in ([40, 41, 42, 43], None):
                    records = list(collect_episodes(batch, first, seeds=seeds))
                    for i, env in enumerate(singles):
                        observation, info = env.reset(seed=None if seeds is None else seeds[i])
                        expected = []
                        while True:
                            action = first(i, observation, info)
                            following, reward, terminated, truncated, following_info = env.step(action)
                            expected.append(Transition(i, observation, action, reward, following,
                                                       terminated, truncated, info, following_info))
                            observation, info = following, following_info
                            if terminated or truncated:
                                break
                        actual = [record for record in records if record.member == i]
                        self.assertEqual(plain(actual), plain(expected))
                        self.assertTrue(actual[-1].terminated or actual[-1].truncated)
                        self.assertEqual(actual[-1].next_info["episode_index"], 0 if seeds else 1)
                        self.assertEqual(len(actual), actual[-1].next_info["learner_steps"])

    def test_records_and_policy_inputs_have_independent_storage(self):
        env = GameEnv("tictactoe", payloads=("cells",))
        batch = GameEnvBatch([env])
        try:
            def destructive_policy(member, observation, info):
                action = first(member, observation, info)
                observation["cells"][:] = 99
                info["observer"] = 99
                return action
            records = list(collect_episodes(batch, destructive_policy, seeds=[42]))
            saved = plain(records)
            self.assertTrue(all(record.info["observer"] == 0 for record in records))
            self.assertTrue(all(np.isin(record.observation["cells"], [-1, 0, 1]).all()
                                for record in records))
            records[0].next_observation["cells"][:] = 77
            self.assertFalse((records[1].observation["cells"] == 77).any())
            records[0].next_observation["cells"][:] = saved[0]["next_observation"]["cells"]
            batch.reset(seeds=[43])
            batch.close()
            env.close()
            self.assertEqual(plain(records), saved)
        finally:
            batch.close()
            env.close()

    def test_successful_peer_is_delivered_before_member_failure(self):
        with ExitStack() as stack:
            envs = [GameEnv("tictactoe", payloads=("cells",)) for _ in range(2)]
            for env in envs:
                stack.callback(env.close)
            batch = stack.enter_context(GameEnvBatch(envs))
            original = envs[0]._format_info
            count = 0
            def fail_on_step(*args):
                nonlocal count
                count += 1
                if count == 2:
                    raise MemoryError("injected collection step output failure")
                return original(*args)
            with patch.object(envs[0], "_format_info", side_effect=fail_on_step):
                iterator = collect_episodes(batch, first, seeds=[5, 6])
                record = next(iterator)
                self.assertEqual(record.member, 1)
                with self.assertRaises(CollectionError) as caught:
                    next(iterator)
            self.assertIn("MemoryError", caught.exception.errors[0])
            self.assertEqual(caught.exception.statuses, ("failed", "ready"))

    def test_reset_error_and_invalid_policy_produce_no_records(self):
        with ExitStack() as stack:
            env = GameEnv("tictactoe", payloads=("cells",))
            stack.callback(env.close)
            batch = stack.enter_context(GameEnvBatch([env]))
            with patch.object(env, "_format_info", side_effect=MemoryError("reset output")):
                with self.assertRaises(CollectionError):
                    next(collect_episodes(batch, first, seeds=[42]))
            self.assertEqual(env.status, "new")
            for action, error in ((99, ValueError), (None, TypeError)):
                with self.assertRaises(error):
                    next(collect_episodes(batch, lambda *args: action, seeds=[42]))
            self.assertEqual(env.status, "ready")
            records = list(collect_episodes(batch, first, seeds=[42]))
            self.assertTrue(records[-1].terminated)


class TabularTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(game="tictactoe", parameters={}, learner=0, opponent="random",
                         opponent_settings={}, max_episode_steps=None)
        self.info = dict(identity={"game": "tictactoe", "encoding_version": 1},
                         observer=0, opponent="random", opponent_settings={})
        self.observation = dict(cells=np.zeros(9), action_mask=np.ones(9, dtype=np.int8))
        self.following = dict(cells=np.array([1, -1, 0, 0, 0, 0, 0, 0, 0]),
                              action_mask=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8))
        self.transition = Transition(0, self.observation, 0, 0, self.following,
                                     False, False, self.info, self.info)

    def test_exact_update_uses_legal_continuation_and_bootstraps_truncation(self):
        for truncated in (False, True):
            table = QTable(self.task, alpha=0.5, gamma=0.9)
            table.values[(0,)*9] = np.array([0.2]+[0.0]*8)
            table.values[table._key(self.following)] = np.array([999, 999, 0.8]+[0.1]*6)
            transition = replace(self.transition, reward=0.1, truncated=truncated)
            table.update(transition)
            self.assertAlmostEqual(table.values[(0,)*9][0], 0.51)
            self.assertEqual(np.count_nonzero(table.values[(0,)*9]), 1)

    def test_terminal_target_is_final_reward_with_empty_legal_mask(self):
        table = QTable(self.task, alpha=0.5, gamma=0.9)
        table.values[(0,)*9] = np.array([0.2]+[0.0]*8)
        final = dict(cells=self.following["cells"], action_mask=np.zeros(9, dtype=np.int8))
        table.update(replace(self.transition, reward=-1, terminated=True, next_observation=final))
        self.assertAlmostEqual(table.values[(0,)*9][0], -0.4)

    def test_exploration_and_greedy_choices_are_legal_and_evaluation_keeps_table(self):
        table = QTable(self.task)
        table.values[table._key(self.following)] = np.array([999, 999, 0.8]+[0.1]*6)
        for epsilon in (0, 1):
            rng = np.random.Generator(np.random.PCG64(42))
            actions = [table.choose(self.following, self.info, rng, epsilon=epsilon) for _ in range(40)]
            self.assertTrue(all(action in range(2, 9) for action in actions))
            if epsilon == 0:
                self.assertEqual(set(actions), {2})
            else:
                self.assertGreater(len(set(actions)), 1)
        self.assertEqual(len(table.values), 1)
        with self.assertRaises(ValueError):
            table.choose(self.following, dict(self.info, observer=2), rng)

    def test_training_save_reload_and_seeded_evaluation_match_for_both_seats(self):
        for learner in (0, 1):
            task = dict(self.task, learner=learner)
            tables = [QTable(task), QTable(task)]
            results = [EXAMPLE.run(table, rounds=12, members=4, seed=42, epsilon=0.3, learning=True)
                       for table in tables]
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0]["episodes"], 48)
            self.assertTrue(any(np.count_nonzero(row) for row in tables[0].values.values()))
            with tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory)/f"table{i}.json" for i in range(2)]
                for table, path in zip(tables, paths):
                    table.save(path)
                self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
                loaded = QTable.load(paths[0], task=task)
                self.assertEqual(tables[0].identity, loaded.identity)
                self.assertEqual(tables[0].values.keys(), loaded.values.keys())
                for key in loaded.values:
                    np.testing.assert_array_equal(tables[0].values[key], loaded.values[key])
                before = paths[0].read_bytes()
                evaluation = [EXAMPLE.run(table, rounds=3, members=4, seed=800, epsilon=0,
                                          learning=False, learner=learner)
                              for table in (tables[0], loaded)]
                self.assertEqual(evaluation[0], evaluation[1])
                loaded.save(paths[1])
                self.assertEqual(paths[1].read_bytes(), before)

    def test_saved_table_rejects_task_version_nonfinite_and_duplicate_rows(self):
        table = QTable(self.task)
        table.update(self.transition)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"table.json"
            table.save(path)
            original = json.loads(path.read_text())
            with self.assertRaises(ValueError):
                QTable.load(path, task=dict(self.task, opponent="osla"))
            for change in (lambda value: value.update(version=99),
                           lambda value: value["rows"][0]["values"].__setitem__(0, float("nan")),
                           lambda value: value["rows"].append(value["rows"][0])):
                value = json.loads(json.dumps(original))
                change(value)
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    QTable.load(path, task=self.task)

    def test_evaluation_accepts_other_opponents_and_training_retains_its_task(self):
        table = QTable(self.task)
        other = dict(self.info, opponent="osla")
        rng = np.random.Generator(np.random.PCG64(42))
        self.assertIn(table.choose(self.observation, other, rng, epsilon=0), range(9))
        with self.assertRaises(ValueError):
            table.update(replace(self.transition, info=other, next_info=other))
        with self.assertRaises(ValueError):
            EXAMPLE.run(table, rounds=1, members=1, seed=42, epsilon=0.2,
                        learning=True, opponent="osla")
        self.assertEqual(table.task, self.task)

    def test_evaluation_file_runs_exact_games_against_each_opponent_and_preserves_model(self):
        example = Path(__file__).parents[1] / "examples/evaluate_tabular_q.py"
        with tempfile.TemporaryDirectory() as directory:
            for learner in (0, 1):
                task = dict(self.task, learner=learner)
                table = QTable(task)
                EXAMPLE.run(table, rounds=4, members=2, seed=42, epsilon=0.3, learning=True)
                path = Path(directory)/f"table-seat{learner}.json"
                table.save(path)
                original = path.read_bytes()
                for opponent in (None, "first_available", "random", "osla", "osla_python"):
                    command = [sys.executable, str(example), "--model", str(path),
                               "--games", "13", "--seed", "800"]
                    if opponent:
                        command += ["--opponent", opponent]
                    outputs = [json.loads(subprocess.check_output(command, text=True)) for _ in range(2)]
                    self.assertEqual(outputs[0], outputs[1])
                    result = outputs[0]
                    self.assertEqual(result["episodes"], 13)
                    self.assertEqual(sum(result[name] for name in ("wins", "losses", "draws", "truncated")), 13)
                    self.assertEqual(result["win_rate"], result["wins"]/13)
                    self.assertEqual(result["mean_reward"], (result["wins"]-result["losses"])/13)
                    self.assertEqual(result["training_task"], task)
                    self.assertEqual(result["evaluation_task"], dict(task, opponent=opponent or "random", learner=None))
                    self.assertEqual(result["by_seat"]["0"]["episodes"], 7)
                    self.assertEqual(result["by_seat"]["1"]["episodes"], 6)
                    for key in ("episodes", "wins", "losses", "draws", "truncated", "transitions"):
                        self.assertEqual(result[key], sum(row[key] for row in result["by_seat"].values()))
                    self.assertEqual(path.read_bytes(), original)
                bad = subprocess.run([sys.executable, str(example), "--model", str(path), "--games", "0"],
                                     capture_output=True, text=True)
                self.assertEqual(bad.returncode, 2)

    def test_alternating_training_visits_both_seats_in_logical_game_order(self):
        for members, rounds in ((1, 5), (2, 3), (3, 3)):
            table = QTable(dict(self.task, learner=None))
            starts, updates = [], []
            original_choose, original_update = table.choose, table.update
            def choose(observation, info, rng, **kwargs):
                if info["learner_steps"] == 0:
                    starts.append(info["observer"])
                return original_choose(observation, info, rng, **kwargs)
            def update(transition):
                self.assertEqual(transition.info["observer"], transition.next_info["observer"])
                updates.append(transition.info["observer"])
                original_update(transition)
            with patch.object(table, "choose", side_effect=choose), patch.object(table, "update", side_effect=update):
                result = EXAMPLE.run(table, rounds=rounds, members=members, seed=42, epsilon=0.2, learning=True)
            games = rounds*members
            self.assertEqual(starts, [i % 2 for i in range(games)])
            self.assertEqual(result["by_seat"]["0"]["episodes"], (games+1)//2)
            self.assertEqual(result["by_seat"]["1"]["episodes"], games//2)
            for learner in (0, 1):
                self.assertEqual(updates.count(learner), result["by_seat"][str(learner)]["transitions"])
            self.assertTrue(any(key.count(-1) == key.count(1) for key in table.values))
            self.assertTrue(any(key.count(-1) == key.count(1)+1 for key in table.values))

    def test_alternating_model_reload_repeats_and_fixed_overrides_report_unused_seat(self):
        task = dict(self.task, learner=None)
        tables = [QTable(task), QTable(task)]
        results = [EXAMPLE.run(table, rounds=8, members=3, seed=42, epsilon=0.3, learning=True)
                   for table in tables]
        self.assertEqual(results[0], results[1])
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/f"mixed{i}.json" for i in range(2)]
            for table, path in zip(tables, paths):
                table.save(path)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            saved = json.loads(paths[0].read_text())
            self.assertEqual(saved["version"], 2)
            self.assertIsNone(saved["task"]["learner"])
            loaded = QTable.load(paths[0])
            for learner in (None, 0, 1):
                results = [EXAMPLE.run(table, rounds=5, members=1, seed=800, epsilon=0,
                                       learning=False, learner=learner) for table in (tables[0], loaded)]
                self.assertEqual(results[0], results[1])
                if learner is not None:
                    self.assertEqual(results[0]["by_seat"][str(learner)]["episodes"], 5)
                    unused = results[0]["by_seat"][str(1-learner)]
                    self.assertEqual(unused["episodes"], 0)
                    self.assertIsNone(unused["win_rate"])
                    self.assertIsNone(unused["mean_reward"])
            loaded.save(paths[1])
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            for learner in (0, 1):
                command = [sys.executable, str(Path(__file__).parents[1]/"examples/evaluate_tabular_q.py"),
                           "--model", str(paths[0]), "--games", "5", "--learner", str(learner)]
                result = json.loads(subprocess.check_output(command, text=True))
                self.assertEqual(result["evaluation_task"]["learner"], learner)
                self.assertEqual(result["by_seat"][str(learner)]["episodes"], 5)

    def test_legacy_fixed_model_evaluates_both_seats_and_training_checks_viewpoint(self):
        table = QTable(self.task)
        EXAMPLE.run(table, rounds=4, members=2, seed=42, epsilon=0.2, learning=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"legacy.json"
            table.save(path)
            value = json.loads(path.read_text())
            value["version"] = 1
            path.write_text(json.dumps(value))
            loaded = QTable.load(path, task=self.task)
            result = EXAMPLE.run(loaded, rounds=4, members=1, seed=800, epsilon=0, learning=False)
            self.assertEqual([row["episodes"] for row in result["by_seat"].values()], [2, 2])
            self.assertEqual(loaded.task["learner"], 0)
            value["task"]["learner"] = None
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                QTable.load(path)
        with self.assertRaises(ValueError):
            table.update(replace(self.transition, info=dict(self.info, observer=1),
                                 next_info=dict(self.info, observer=1)))
        mixed = QTable(dict(self.task, learner=None))
        with self.assertRaises(ValueError):
            mixed.update(replace(self.transition, next_info=dict(self.info, observer=1)))
        self.assertFalse(mixed.values)

    def test_training_cli_defaults_to_shared_alternating_table_and_preserves_fixed_option(self):
        example = Path(__file__).parents[1]/"examples/tabular_q.py"
        with tempfile.TemporaryDirectory() as directory:
            for learner in (None, 0, 1):
                path = Path(directory)/f"cli{learner}.json"
                command = [sys.executable, str(example), "--rounds", "3", "--members", "3",
                           "--eval-rounds", "1", "--output", str(path)]
                if learner is not None:
                    command += ["--learner", str(learner)]
                result = json.loads(subprocess.check_output(command, text=True))
                self.assertEqual(result["task"]["learner"], learner)
                self.assertEqual(result["training"]["episodes"], 9)
                for phase, games in (("training", 9), ("baseline", 3), ("reloaded_evaluation", 3)):
                    counts = result[phase]["by_seat"]
                    expected = [(games+1)//2, games//2] if learner is None else [
                        games if seat == learner else 0 for seat in (0, 1)]
                    self.assertEqual([counts[str(seat)]["episodes"] for seat in (0, 1)], expected)
