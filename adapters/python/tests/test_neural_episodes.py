"""Exercise real intermediate/teacher episode operations and existing recovery guarantees."""
from contextlib import ExitStack
from unittest.mock import patch
import unittest
import numpy as np
from tag_rust.gym import GameEnv
from tag_rust.gym_batch import GameEnvBatch


class IntermediateEpisodes(unittest.TestCase):
    def test_capture_preserves_both_game_traces_seats_and_opponent_rng(self):
        for game in ("tictactoe", "connect4"):
            for seat in (0, 1):
                with self.subTest(game=game, seat=seat), ExitStack() as stack:
                    a = stack.enter_context(GameEnv(game, learner=seat, payloads=("cells",), opponent="random"))
                    b = stack.enter_context(GameEnv(game, learner=seat, payloads=("cells",), opponent="random", include_intermediate=True))
                    oa, ia = a.reset(seed=17)
                    ob, ib = b.reset(seed=17)
                    self.assertIsNone(ib["intermediate"])
                    while True:
                        action = int(np.flatnonzero(oa["action_mask"])[0])
                        before = int(ia["progress"][0])
                        oa, ra, ta, xa, ia = a.step(action)
                        ob, rb, tb, xb, ib = b.step(action)
                        for name in oa:
                            np.testing.assert_array_equal(oa[name], ob[name])
                        self.assertEqual((ra, ta, xa, ia["opponent_rng"]), (rb, tb, xb, ib["opponent_rng"]))
                        mid = ib["intermediate"]
                        self.assertEqual(mid["info"]["observer"], seat)
                        self.assertEqual(mid["info"]["progress"][0], before + 1)
                        self.assertFalse(mid["observation"]["action_mask"].any())
                        mid["observation"]["cells"][:] = 99
                        self.assertFalse((ob["cells"] == 99).any())
                        if ta or xa:
                            break

    def test_immediate_terminal_horizon_and_output_bounds(self):
        with GameEnv("tictactoe", {"grid_size":1}, payloads=("cells",), include_intermediate=True) as env:
            env.reset(seed=1)
            _, reward, done, truncated, info = env.step(0)
            self.assertEqual((reward, done, truncated), (1., True, False))
            self.assertIsNone(info["intermediate"]["info"]["next_player"])
            self.assertEqual(info["intermediate"]["info"]["results"][0], 1)
        with GameEnv("connect4", payloads=("cells",), include_intermediate=True, max_episode_steps=1) as env:
            env.reset(seed=1)
            _, _, done, truncated, info = env.step(0)
            self.assertEqual((done, truncated), (False, True))
            self.assertEqual(info["intermediate"]["info"]["next_player"], 1)
        with self.assertRaises(ValueError):
            GameEnv("tictactoe", payloads=("cells",), include_intermediate=True, max_bytes=400)
        with GameEnv("tictactoe", payloads=("cells",), max_bytes=250):
            pass
        with self.assertRaises(ValueError):
            GameEnv("tictactoe", payloads=("cells",), teacher="osla", max_bytes=250)

    def test_mixed_batches_inactive_rows_and_failed_intermediates(self):
        with ExitStack() as stack:
            envs = [stack.enter_context(GameEnv("tictactoe", payloads=("cells",), include_intermediate=flag)) for flag in (True, False)]
            batch = stack.enter_context(GameEnvBatch(envs))
            reset = batch.reset(seeds=[2, 3])
            self.assertFalse(reset.intermediate_valid.any())
            result = batch.step([4, 4])
            np.testing.assert_array_equal(result.intermediate_valid, [True, False])
            self.assertIsNone(result.intermediate_infos[1])
            self.assertFalse(result.intermediate_observations["cells"][1].any())
            with patch.object(envs[0], "_format_info", side_effect=MemoryError("controlled")):
                action = int(np.flatnonzero(result.observations["action_mask"][0])[0])
                failed = batch.step([action, None])
            self.assertFalse(failed.intermediate_valid.any())
            self.assertFalse(failed.valid.any())
            self.assertFalse(failed.intermediate_observations["cells"].any())
            self.assertEqual(batch.statuses[0], "failed")
            batch.reset(indices=[0], seeds=[2])
            with self.assertRaises(MemoryError):
                GameEnvBatch(envs, max_bytes=700)
        with GameEnv("tictactoe", payloads=("cells",)) as env, GameEnvBatch([env]) as batch:
            self.assertIsNone(batch.reset(seeds=[1]).intermediate_observations)


class TeacherEpisodes(unittest.TestCase):
    def test_teacher_replay_separate_seeds_and_capture_parity(self):
        for game in ("tictactoe", "connect4"):
            for seat in (0, 1):
                traces = []
                for capture in (False, True):
                    with GameEnv(game, learner=seat, opponent="osla", teacher="osla", payloads=("cells",), include_intermediate=capture) as env:
                        with self.assertRaises(ValueError):
                            env.reset(seed=31)
                        for repeat in range(2):
                            env.reset(seed=31, options={"teacher_seed":811})
                            trace = []
                            while True:
                                obs, reward, done, truncated, info = env.step_teacher()
                                self.assertEqual(info["teacher_seed"], 811)
                                self.assertEqual(info["teacher_rng"][1], 811)
                                self.assertEqual(info["opponent_rng"][1], info["opponent_seed"])
                                trace.append((info["action"], reward, done, truncated, tuple(obs["cells"]), info["teacher_rng"], info["opponent_rng"]))
                                if done or truncated:
                                    break
                            traces.append(trace)
                for trace in traces[1:]:
                    self.assertEqual(trace, traces[0])

    def test_bulk_teacher_preflight_and_reset_atomicity(self):
        with ExitStack() as stack:
            teacher = stack.enter_context(GameEnv("tictactoe", teacher="osla", include_intermediate=True, payloads=("cells",)))
            ordinary = stack.enter_context(GameEnv("tictactoe", payloads=("cells",)))
            batch = stack.enter_context(GameEnvBatch([teacher, ordinary]))
            initial = batch.reset(seeds=[8, 9], options=[{"teacher_seed":3}, None])
            with self.assertRaises(ValueError):
                batch.step_teacher()
            with self.assertRaises(ValueError):
                batch.reset(seeds=[1, 1], options=[{"teacher_seed":4}, {"teacher_seed":5}])
            result = batch.step_teacher(active=[True, False])
            self.assertEqual(result.infos[0]["underlying_moves"], 2)
            self.assertEqual(result.infos[0]["teacher_rng"][1], 3)
            self.assertTrue(result.intermediate_valid[0])
            self.assertFalse(result.valid[1])
            self.assertEqual(initial.infos[1]["opponent_rng"][3], 0)


if __name__ == "__main__":
    unittest.main()
