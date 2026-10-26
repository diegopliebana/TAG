"""Check complete groups, episode streams, batch policy calls and retained trajectory data."""
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch
import importlib.util
import unittest
import numpy as np
import tag_rust as tag
from tag_rust.neural_seeds import episode_seed
from tag_rust.collection import collect_episodes
from tag_rust.gym import GameEnv
from tag_rust.gym_batch import GameEnvBatch

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.neural import PolicyValueNetwork
    from tag_rust.neural_inputs import ModelContract
    from tag_rust.neural_snapshot import FrozenModel
    from tag_rust.neural_collection import collect_episode_group, GroupCollectionError


class CollectionStreams(unittest.TestCase):
    def test_exact_s1_seed_fixtures_and_purpose_separation(self):
        self.assertEqual(episode_seed(42,"actor_critic",0,"reset"), 4314567813735656003)
        self.assertEqual(episode_seed(42,"actor_critic",1,"reset"), 15786328513068299086)
        self.assertEqual(episode_seed(42,"teacher",0,"teacher_learner"), 800940763950071329)
        values = {episode_seed(42,phase,0,purpose) for phase in ("teacher", "actor_critic", "evaluation")
                  for purpose in ("reset", "policy_sampling", "teacher_learner")}
        self.assertEqual(len(values), 9)
        with self.assertRaises(TypeError):
            episode_seed(True,"teacher",0,"reset")

    def test_bulk_policy_gets_active_rows_once_and_bad_count_applies_no_action(self):
        with ExitStack() as stack:
            envs = [stack.enter_context(GameEnv("tictactoe", learner=i, payloads=("cells",), max_episode_steps=1)) for i in (0,1)]
            batch = stack.enter_context(GameEnvBatch(envs))
            calls = []
            def choose(members, observations, infos):
                calls.append(tuple(members))
                return [int(np.flatnonzero(o["action_mask"])[0]) for o in observations]
            rows = list(collect_episodes(batch, choose_batch=choose, seeds=[4,5]))
            self.assertEqual(calls, [(0,1)])
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row.truncated or row.terminated for row in rows))
            with self.assertRaises(ValueError):
                list(collect_episodes(batch, choose_batch=lambda *args:[], seeds=[4,5]))
            self.assertEqual(batch.statuses, ("ready", "ready"))


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class NeuralGroups(unittest.TestCase):
    def snapshot(self, name):
        with tag.create(name) as game:
            model = PolicyValueNetwork(ModelContract.from_spec(game.spec(),payloads=("cells",)), (2,))
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            return FrozenModel(model)

    def test_both_games_complete_teacher_neural_groups_and_retained_intermediates(self):
        for name in ("tictactoe", "connect4"):
            snapshot = self.snapshot(name)
            for phase in ("teacher", "actor_critic"):
                with self.subTest(game=name, phase=phase):
                    group = collect_episode_group(name, episode_count=3, master_seed=42, phase=phase,
                        first_episode_id=3, snapshot=None if phase=="teacher" else snapshot, payloads=("cells",))
                    self.assertEqual([e.seat for e in group.episodes], [1,0,1])
                    self.assertEqual(group.next_episode_id, 6)
                    self.assertEqual(group.outcomes["combined"]["episodes"], 3)
                    self.assertEqual(group.outcomes["per_seat"][0]["episodes"], 1)
                    for episode in group.episodes:
                        self.assertTrue(episode.steps[-1].transition.terminated)
                        for index, step in enumerate(episode.steps):
                            self.assertEqual(step.index, index)
                            row = step.transition
                            self.assertEqual(row.info["observer"], episode.seat)
                            self.assertEqual(row.intermediate["info"]["observer"], episode.seat)
                            self.assertEqual(row.intermediate["info"]["progress"][0], row.info["progress"][0]+1)
                            self.assertEqual(row.next_info["observer"], episode.seat)
                            self.assertTrue(row.observation["action_mask"][row.action])
                            self.assertFalse(np.shares_memory(row.observation["cells"], row.next_observation["cells"]))
                            if phase=="teacher":
                                self.assertIsNone(step.policy_statistics)
                                self.assertEqual(row.next_info["action"], row.action)
                            else:
                                self.assertEqual(step.policy_statistics["model_identity"], snapshot.identity)
                                self.assertAlmostEqual(step.policy_statistics["value"],0.)

    def test_seeded_groups_repeat_evaluation_isolated_fixed_seats_and_truncation(self):
        snapshot = self.snapshot("tictactoe")
        args = dict(episode_count=3, master_seed=39, snapshot=snapshot, max_episode_steps=1)
        first = collect_episode_group("tictactoe", **args)
        with patch.object(snapshot, "predict", wraps=snapshot.predict) as predict:
            collect_episode_group("tictactoe", phase="evaluation", learner=1, **args)
            self.assertEqual(predict.call_count, 1)
        again = collect_episode_group("tictactoe", **args)
        for a,b in zip(first.episodes,again.episodes):
            self.assertEqual(a.reset_seed,b.reset_seed)
            self.assertEqual(a.steps[0].transition.action,b.steps[0].transition.action)
            np.testing.assert_array_equal(a.steps[-1].transition.next_observation["cells"],b.steps[-1].transition.next_observation["cells"])
            self.assertTrue(a.steps[-1].transition.truncated)
            self.assertTrue(a.steps[-1].transition.next_observation["action_mask"].any())
        fixed = collect_episode_group("tictactoe", learner=1, **args)
        self.assertEqual([e.seat for e in fixed.episodes],[1,1,1])
        self.assertEqual(fixed.outcomes["per_seat"][0]["episodes"],0)
        # The same logical episode has its own stream even in a smaller group.
        single = collect_episode_group("tictactoe", episode_count=1, first_episode_id=2,
                  master_seed=39, snapshot=snapshot, max_episode_steps=1)
        self.assertEqual(single.episodes[0].steps[0].transition.action, first.episodes[2].steps[0].transition.action)

    def test_partial_group_failure_retains_diagnostics_and_successful_records(self):
        snapshot = self.snapshot("tictactoe")
        from tag_rust.neural_play import NeuralPolicy
        original = NeuralPolicy.choose_batch
        calls = 0
        def fail_later(policy, *args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("controlled policy failure")
            return original(policy,*args)
        with patch.object(NeuralPolicy,"choose_batch",fail_later):
            with self.assertRaises(GroupCollectionError) as caught:
                collect_episode_group("tictactoe",episode_count=2,master_seed=1,snapshot=snapshot)
        self.assertTrue(all(len(e.steps)==1 for e in caught.exception.partial_episodes))
        self.assertIn("controlled policy failure",str(caught.exception.cause))


if __name__ == "__main__":
    unittest.main()
