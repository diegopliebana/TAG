"""Exact bulk/individual parity, failure isolation and explicit-reset ownership."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from tag_rust.gym import GameEnv
from tag_rust.gym_batch import GameEnvBatch


def same_info(test,a,b):
    test.assertEqual(a.keys(),b.keys())
    for key in a:
        if isinstance(a[key],np.ndarray):np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        else:test.assertEqual(a[key],b[key],key)


class EpisodeBatchTests(unittest.TestCase):
    def test_every_transition_and_rng_matches_individual_mixed_members(self):
        cases=[("tictactoe",{}),("tictactoe",{"grid_size":5}),
               ("connect4",{}),("connect4",{"width":4,"height":3,"winning_length":3})]
        for name,params in cases:
            settings=[dict(learner=i%2,opponent=policy,payloads=("cells","board"),
                           max_episode_steps=3 if i%3==0 else None)
                      for i,policy in enumerate(("random","osla","first_available","osla_python")*2)]
            originals=[GameEnv(name,params,**s) for s in settings]
            singles=[GameEnv(name,params,**s) for s in settings]
            batch=GameEnvBatch(originals)
            try:
                result=batch.reset(seeds=list(range(20,28)))
                reference=[env.reset(seed=20+i) for i,env in enumerate(singles)]
                self.assertTrue(result.valid.all())
                active=[True]*8
                while any(active):
                    for i,enabled in enumerate(active):
                        if enabled:
                            obs,info=reference[i]
                            for key in obs:np.testing.assert_array_equal(result.observations[key][i],obs[key])
                            same_info(self,result.infos[i],info)
                    actions=[int(np.flatnonzero(reference[i][0]["action_mask"])[0]) if active[i] else None for i in range(8)]
                    old_active=active[:]
                    expected=[]
                    for i,action in enumerate(actions):
                        expected.append(None if action is None else singles[i].step(action))
                    result=batch.step(actions)
                    self.assertEqual(result.valid.tolist(),old_active)
                    self.assertTrue(all(e is None for e in result.errors))
                    for i,transition in enumerate(expected):
                        if transition is None:
                            self.assertIsNone(result.infos[i])
                            for values in result.observations.values():self.assertFalse(values[i].any())
                            continue
                        obs,reward,term,trunc,info=transition
                        for key in obs:np.testing.assert_array_equal(result.observations[key][i],obs[key])
                        same_info(self,result.infos[i],info)
                        self.assertEqual((result.rewards[i],result.terminated[i],result.truncated[i]),(reward,term,trunc))
                        reference[i]=(obs,info)
                        active[i]=not(term or trunc)
                reset=batch.reset(indices=[6,1],seeds=[None,501])
                for i,seed in [(6,None),(1,501)]:
                    obs,info=singles[i].reset(seed=seed)
                    same_info(self,reset.infos[i],info)
                    for key in obs:np.testing.assert_array_equal(reset.observations[key][i],obs[key])
                self.assertEqual(reset.valid.tolist(),[False,True,False,False,False,False,True,False])
            finally:
                batch.close()
                for env in originals+singles:env.close()

    def test_whole_call_validation_preserves_all_members_and_seeds(self):
        envs=[GameEnv("tictactoe",opponent="random") for _ in range(2)]
        with GameEnvBatch(envs) as batch:
            before=batch.reset(seeds=[1,2])
            states=[deepcopy(e.np_random.bit_generator.state) for e in envs]
            for actions in ([0,99],[0,True],[0],[0,1.2]):
                with self.assertRaises((ValueError,TypeError)):batch.step(actions)
            for kwargs in (dict(indices=[0,0]),dict(indices=[2]),dict(seeds=[1]),dict(seeds=[1,-1]),dict(seeds=[1,False])):
                with self.assertRaises((ValueError,TypeError)):batch.reset(**kwargs)
            for env,state in zip(envs,states):self.assertEqual(env.np_random.bit_generator.state,state)
            after=batch.step([0,0])
            self.assertEqual([i["underlying_moves"] for i in after.infos],[2,2])
            self.assertTrue(before.valid.all())
        for env in envs:env.close()

    def test_input_masks_limits_compatibility_and_lifetime(self):
        a=GameEnv("tictactoe",opponent="first_available")
        b=GameEnv("connect4")
        c=GameEnv("tictactoe",payloads=("cells",))
        for members in ([],[a,a],[a,b],[a,c]):
            with self.assertRaises(ValueError):GameEnvBatch(members)
        with self.assertRaises(MemoryError):GameEnvBatch([a],max_bytes=1)
        batch=GameEnvBatch([a])
        result=batch.reset(seeds=[1]);old=deepcopy(result.observations)
        inactive=batch.step([999],active=[False])
        self.assertFalse(inactive.valid.any());self.assertEqual(inactive.statuses,("ready",))
        for bad in ([1],[True,False]):
            with self.assertRaises(ValueError):batch.step([0],active=bad)
        result.observations["board"][:]=42
        batch.step([4]);batch.close();batch.close()
        self.assertEqual(a.status,"ready")
        a.step(1)
        with self.assertRaises(RuntimeError):batch.step([2])
        self.assertFalse((old["board"]==42).any())
        a.close();b.close();c.close()

    def test_step_format_failure_is_per_member_and_requires_reset(self):
        envs=[GameEnv("tictactoe",opponent="first_available") for _ in range(3)]
        with GameEnvBatch(envs) as batch:
            batch.reset(seeds=[1,2,3])
            with patch.object(envs[1],"_format_info",side_effect=ValueError("injected")):
                result=batch.step([4,4,4])
            self.assertEqual(result.valid.tolist(),[True,False,True])
            self.assertEqual(result.statuses,("ready","failed","ready"))
            self.assertIn("injected",result.errors[1]);self.assertIsNone(result.infos[1])
            for values in result.observations.values():self.assertFalse(values[1].any())
            with self.assertRaises(RuntimeError):batch.step([1,1,1])
            self.assertEqual(batch.step([1,None,1]).valid.tolist(),[True,False,True])
            reset=batch.reset(indices=[1])
            self.assertEqual(reset.valid.tolist(),[False,True,False])
            self.assertEqual(envs[1].status,"ready")
        for env in envs:env.close()

    def test_reset_seed_staging_failure_is_isolated_without_advancing_its_generator(self):
        envs=[GameEnv("tictactoe",opponent="first_available") for _ in range(2)]
        with GameEnvBatch(envs) as batch:
            batch.reset(seeds=[1,2]);batch.step([4,4])
            state=deepcopy(envs[1].np_random.bit_generator.state)
            with patch.object(envs[1],"_prepare_seed",side_effect=MemoryError("seed staging")):
                result=batch.reset()
            self.assertEqual(result.valid.tolist(),[True,False])
            self.assertIn("seed staging",result.errors[1])
            self.assertEqual(envs[1].np_random.bit_generator.state,state)
            with self.assertRaises(ValueError):envs[1].step(4)
            envs[1].step(1)
        for env in envs:env.close()

    def test_reset_format_failure_preserves_old_board_and_seed_progression(self):
        envs=[GameEnv("tictactoe",opponent="first_available") for _ in range(2)]
        with GameEnvBatch(envs) as batch:
            batch.reset(seeds=[1,2]);batch.step([4,4])
            old=deepcopy(envs[1].np_random.bit_generator.state)
            with patch.object(envs[1],"_format_info",side_effect=MemoryError("injected reset")):
                result=batch.reset()
            self.assertEqual(result.valid.tolist(),[True,False])
            self.assertEqual(envs[1].np_random.bit_generator.state,old)
            self.assertEqual(envs[1].status,"ready")
            with self.assertRaises(ValueError):envs[1].step(4)
            envs[1].step(1)
        for env in envs:env.close()

    def test_aggregate_format_failure_marks_all_completed_steps_failed(self):
        envs=[GameEnv("connect4",opponent="random") for _ in range(2)]
        with GameEnvBatch(envs) as batch:
            batch.reset(seeds=[1,2])
            with patch.object(batch,"_format",side_effect=MemoryError("aggregate")):
                with self.assertRaises(MemoryError):batch.step([0,0])
            self.assertEqual(batch.statuses,("failed","failed"))
            batch.reset();self.assertTrue(batch.step([0,0]).valid.all())
        for env in envs:env.close()

    def test_foreign_stale_and_consumed_native_reset_candidates(self):
        a=GameEnv("tictactoe",opponent="first_available")
        b=GameEnv("tictactoe",opponent="first_available")
        with GameEnvBatch([a,b]) as batch,GameEnvBatch([a,b]) as other:
            batch.reset(seeds=[1,2])
            candidate=batch._native.prepare_resets([0,1],[7,8])
            with self.assertRaises(ValueError):other._native.commit_resets(candidate,[0,1])
            a.step(4)
            with self.assertRaises(ValueError):batch._native.commit_resets(candidate,[0,1])
            # The valid second member was not committed before the stale first.
            result=b.step(4)
            self.assertEqual(result[4]["opponent_rng"][1],int(b._seed_info["opponent_seed"]))
            self.assertEqual(result[4]["underlying_moves"],2)
            candidate=batch._native.prepare_resets([0],[7])
            batch._native.commit_resets(candidate,[0])
            with self.assertRaises(ValueError):batch._native.commit_resets(candidate,[0])
        a.close();b.close()

    def test_lock_conflicts_and_closed_selected_members_are_preflight_errors(self):
        a=GameEnv("connect4",opponent="random");b=GameEnv("connect4",opponent="random")
        with GameEnvBatch([a,b]) as batch:
            batch.reset(seeds=[1,2])
            with a._exclusive():
                with self.assertRaises(RuntimeError):batch.step([0,0])
            b.close()
            with self.assertRaises(RuntimeError):batch.step([0,0])
            result=a.step(0)
            self.assertEqual(result[4]["underlying_moves"],2)
            self.assertEqual(batch.step([1,None]).valid.tolist(),[True,False])
        a.close()

    def test_returned_arrays_survive_steps_resets_and_original_close(self):
        envs=[GameEnv("tictactoe",opponent="first_available",payloads=("cells",)) for _ in range(2)]
        batch=GameEnvBatch(envs)
        initial=batch.reset(seeds=[1,2])
        step=batch.step([4,4])
        self.assertTrue((initial.observations["cells"]==0).all())
        expected=step.observations["cells"].copy()
        newer=batch.reset()
        newer.observations["cells"][:]=99
        np.testing.assert_array_equal(step.observations["cells"],expected)
        batch.close()
        for env in envs:env.close()
        np.testing.assert_array_equal(step.observations["cells"],expected)
        self.assertTrue((initial.observations["cells"]==0).all())

    def test_maximum_distinct_member_count_and_partial_initial_reset(self):
        envs=[GameEnv("tictactoe",opponent="random",payloads=("cells",)) for _ in range(256)]
        try:
            with GameEnvBatch(envs) as batch:
                result=batch.reset(indices=[255,0],seeds=[255,0])
                self.assertEqual(result.observations["cells"].shape,(256,9))
                self.assertEqual(np.flatnonzero(result.valid).tolist(),[0,255])
                self.assertEqual(result.statuses[1],"new")
                step=batch.step([0]+[None]*254+[0])
                self.assertEqual(np.flatnonzero(step.valid).tolist(),[0,255])
        finally:
            for env in envs:env.close()

    def test_empty_selection_has_no_side_effects_and_bounds_member_count(self):
        env=GameEnv("tictactoe",opponent="random")
        with GameEnvBatch([env]) as batch:
            out=batch.reset(indices=[],seeds=[])
            self.assertFalse(out.valid.any());self.assertEqual(env.status,"new")
            self.assertFalse(batch.step([None]).valid.any())
            with self.assertRaises(ValueError):GameEnvBatch([env]*257)
        env.close()


if __name__=="__main__":unittest.main()
