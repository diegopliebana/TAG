"""Episode lifecycle and native parity, plus optional real Gymnasium checks."""
from copy import deepcopy
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
from tag_rust import _engine

HAS_GYM = importlib.util.find_spec("gymnasium") is not None
if HAS_GYM:
    from tag_rust.gym import GameEnv
    from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker
    from gymnasium.utils.env_checker import check_env


def config(**kwargs):
    result = dict(game="tictactoe", parameters={}, learner=0,opponent="first_available",
                  opponent_settings={},payloads=["cells","board"],max_episode_steps=None,
                  max_cells=4096,max_bytes=64*1024*1024)
    result.update(kwargs)
    return result


def reset_native(game,seed=11):
    candidate=game.prepare_reset(seed)
    output=candidate.output
    game.commit_reset(candidate)
    return output


class NativeEpisodeTests(unittest.TestCase):
    def test_native_registry_both_games_and_seats(self):
        for name in ("tictactoe","connect4"):
            for learner in (0,1):
                for opponent in ("first_available","random","osla","osla_python"):
                    with self.subTest(game=name,learner=learner,opponent=opponent):
                        game=_engine.Episode(config(game=name,learner=learner,opponent=opponent))
                        out=reset_native(game)
                        self.assertEqual(out["underlying_moves"],learner)
                        while not out["terminated"]:
                            self.assertEqual(out["next_player"],learner)
                            out=game.step(int(out["offered_action_ids"][0]))
                        self.assertEqual(out["observer"],learner)
                        self.assertFalse(out["truncated"])
                        self.assertFalse(out["action_mask"].any())
                        self.assertEqual(game.status,"ended")
                        with self.assertRaises(ValueError):game.step(0)
                        game.close()

    def test_reset_is_staged_and_candidates_are_bound_to_owner(self):
        game=_engine.Episode(config())
        other=_engine.Episode(config())
        old=reset_native(game)
        after=game.step(4)
        candidate=game.prepare_reset(12)
        with self.assertRaises(ValueError):other.commit_reset(candidate)
        self.assertEqual(game.status,"ready")
        # Preparing a reset did not replace the center placement.
        with self.assertRaises(ValueError):game.step(4)
        np.testing.assert_array_equal(old["payloads"]["cells"],np.zeros(9))
        game.commit_reset(candidate)
        with self.assertRaises(ValueError):game.commit_reset(candidate)
        fresh=game.step(4)
        np.testing.assert_array_equal(after["payloads"]["cells"],fresh["payloads"]["cells"])

    def test_lifecycle_horizon_and_input_rejection(self):
        game=_engine.Episode(config(max_episode_steps=1))
        with self.assertRaises(RuntimeError):game.step(0)
        out=reset_native(game)
        for action in (99,):
            with self.assertRaises(ValueError):game.step(action)
            self.assertEqual(game.status,"ready")
        out=game.step(4)
        self.assertTrue(out["truncated"]);self.assertFalse(out["terminated"])
        self.assertTrue(out["action_mask"].any())
        with self.assertRaises(ValueError):game.step(2)
        reset_native(game)
        game.fail()
        with self.assertRaises(RuntimeError):game.step(4)
        reset_native(game)
        self.assertEqual(game.status,"ready")
        game.close();game.close()
        with self.assertRaises(RuntimeError):game.prepare_reset(1)

    def test_explicit_draw_and_both_learner_results(self):
        game=_engine.Episode(config())
        reset_native(game)
        for action in [4,1,6,5,8]:
            out=game.step(action)
        self.assertEqual((out["reward"],out["terminated"],out["results"].tolist()),(0.,True,[2,2]))
        for learner in (0,1):
            game=_engine.Episode(config(learner=learner))
            out=reset_native(game)
            while not out["terminated"]:
                out=game.step(int(out["offered_action_ids"][0]))
            self.assertEqual(out["reward"],1. if learner==0 else -1.)

    def test_configuration_restrictions_and_bounds(self):
        for args in (dict(opponent="python"),dict(opponent_settings={"seed":4}),
                     dict(parameters={"grid_size":1},learner=1),dict(max_episode_steps=0),
                     dict(payloads=["image"]),dict(max_bytes=1),
                     dict(opponent="osla_python",opponent_settings={"max_candidates":8})):
            with self.subTest(args=args),self.assertRaises(ValueError):_engine.Episode(config(**args))
        spec=_engine.Episode(config()).spec()
        self.assertEqual(spec["payloads"][0]["tensor"]["bounds"],(-1,1))
        self.assertEqual(spec["payloads"][1]["tensor"]["bounds"],(0,1))
        game=_engine.Episode(config(parameters={"grid_size":1},max_episode_steps=1))
        reset_native(game);out=game.step(0)
        self.assertEqual((out["reward"],out["terminated"],out["truncated"]),(1.,True,False))


@unittest.skipUnless(HAS_GYM,"Gymnasium is not installed; install the documented optional dependency")
class GymEpisodeTests(unittest.TestCase):
    def test_spaces_passive_checks_and_complete_masked_games(self):
        cases=[("tictactoe",{}),("tictactoe",{"grid_size":4}),
               ("connect4",{}),("connect4",{"width":4,"height":3,"winning_length":3})]
        for name,parameters in cases:
            for learner in (0,1):
                for policy in ("first_available","random","osla","osla_python"):
                    env=GameEnv(name,parameters,learner=learner,opponent=policy,payloads=("cells","board"))
                    obs,info=env_reset_passive_checker(env,seed=42)
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(info["observer"],learner)
                    previous=deepcopy(obs)
                    env.action_space.seed(4)
                    while True:
                        action=env.action_space.sample(mask=obs["action_mask"])
                        obs,reward,term,trunc,info=env_step_passive_checker(env,action)
                        self.assertTrue(env.observation_space.contains(obs))
                        self.assertEqual(info["observer"],learner)
                        if term or trunc:break
                    self.assertFalse(trunc)
                    self.assertFalse(obs["action_mask"].any())
                    self.assertIn(reward,(-1.,0.,1.))
                    env.close()
                    self.assertTrue(env.observation_space.contains(previous))

    def test_reseed_and_successive_reset_reproduce_outputs_and_rng(self):
        for learner in (0,1):
            env=GameEnv("connect4",learner=learner,opponent="random",payloads=("cells",))
            def run():
                obs,info=env.reset(seed=845)
                first=deepcopy((obs,info))
                obs,reward,term,trunc,info=env.step(int(np.flatnonzero(obs["action_mask"])[0]))
                step=deepcopy((obs,info,reward,term,trunc))
                second=env.reset()
                return first,step,second
            a,b=run(),run()
            for (obs_a,info_a,*tail_a),(obs_b,info_b,*tail_b) in zip(a,b):
                for key in obs_a:np.testing.assert_array_equal(obs_a[key],obs_b[key])
                for key in ("opponent_rng","episode_index","environment_seed","opponent_seed","underlying_moves"):
                    self.assertEqual(info_a[key],info_b[key])
                self.assertEqual(tail_a,tail_b)
            self.assertNotEqual(a[0][1]["opponent_seed"],a[2][1]["opponent_seed"])
            env.close()

    def test_output_failures_preserve_reset_but_poison_step_and_recover(self):
        env=GameEnv("tictactoe",opponent="first_available",payloads=("cells",))
        env.reset(seed=5)
        rng=deepcopy(env.np_random.bit_generator.state)
        with patch.object(env,"_format",side_effect=MemoryError("injected output failure")):
            with self.assertRaises(MemoryError):env.reset()
        self.assertEqual(env.np_random.bit_generator.state,rng)
        self.assertEqual(env.status,"ready")
        env.step(4)
        with patch.object(env,"_format",side_effect=MemoryError("injected output failure")):
            with self.assertRaises(MemoryError):env.step(1)
        self.assertEqual(env.status,"failed")
        with self.assertRaises(RuntimeError):env.step(2)
        with patch.object(env,"_format",side_effect=MemoryError("injected output failure")):
            with self.assertRaises(MemoryError):env.reset()
        self.assertEqual(env.status,"failed")
        env.reset();self.assertEqual(env.status,"ready")
        env.close()

    def test_input_errors_are_retryable_and_arrays_are_independent(self):
        env=GameEnv("tictactoe",opponent="first_available",payloads=("cells","board"))
        old,info=env.reset(seed=6)
        for action in (True,np.bool_(False),1.2,-1,100):
            with self.assertRaises((TypeError,ValueError,OverflowError)):env.step(action)
            self.assertEqual(env.status,"ready")
        new,*_=env.step(4)
        with self.assertRaises(ValueError):env.step(4)
        self.assertEqual(env.status,"ready")
        self.assertTrue((old["cells"]==0).all())
        new["cells"][:]=42;new["action_mask"][:]=0
        obs,*_=env.step(1)
        self.assertTrue(env.observation_space.contains(obs))
        env.close()
        self.assertTrue((old["cells"]==0).all())

    def test_horizon_terminal_and_unmodified_checker_smoke(self):
        env=GameEnv("tictactoe",opponent="first_available",max_episode_steps=1)
        env.reset(seed=1)
        obs,r,term,trunc,info=env.step(4)
        self.assertEqual((r,term,trunc),(0.,False,True));self.assertTrue(obs["action_mask"].any())
        env.close()
        env=GameEnv("tictactoe",{"grid_size":1},opponent="random",max_episode_steps=1)
        env.reset(seed=1)
        _,r,term,trunc,_=env.step(0)
        self.assertEqual((r,term,trunc),(1.,True,False));env.close()
        # Every initial action is legal and ends this smoke configuration.
        env=GameEnv("tictactoe",{"grid_size":1},opponent="random")
        check_env(env,skip_render_check=True)
        env.close()

    def test_invalid_options_concurrency_and_action_rng(self):
        env=GameEnv("connect4",opponent="random")
        env.reset(seed=1)
        env.action_space.seed(3);state=deepcopy(env.action_space.np_random.bit_generator.state)
        env.reset(seed=1)
        self.assertEqual(state,env.action_space.np_random.bit_generator.state)
        with self.assertRaises(ValueError):env.reset(options={"unsupported":1})
        with env._exclusive():
            with self.assertRaises(RuntimeError):env.step(0)
        env.close()


if __name__=="__main__":unittest.main()
