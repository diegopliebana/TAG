"""Numeric learning semantics, real parameter updates and exact checkpoint continuation."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tag_rust as tag

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.learning.training import (TrainingSession, TrainingSettings, discounted_returns,
                                             make_targets, loss_terms, LearningTargets)
    from tag_rust.learning.artifacts import (save_model, load_model, model_document, load_document,
                                             save_checkpoint, load_checkpoint)
    from tag_rust.learning.collection import collect_episode_group
    from tag_rust.learning.inputs import ModelContract
    from tag_rust.learning.snapshot import FrozenModel
    from tag_rust.learning.evaluation import evaluate_model


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class NeuralTraining(unittest.TestCase):
    def session(self, game="tictactoe", **kw):
        kw.setdefault("master_seed", 42)
        return TrainingSession(TrainingSettings(game=game, teacher_episodes=3, actor_episodes=5,
                                                group_size=2, hidden_sizes=(4,), **kw))

    def test_returns_numeric_fixtures_and_invalid_inputs(self):
        np.testing.assert_allclose(discounted_returns([0,0,1]), [.9801,.99,1])
        np.testing.assert_allclose(discounted_returns([0], continuation=.8), [.792])
        np.testing.assert_allclose(discounted_returns([0,0,-1]), [-.9801,-.99,-1])
        for rewards in ([], [float("nan")], [[1]]):
            with self.assertRaises(ValueError):
                discounted_returns(rewards)

    def test_frozen_bootstrap_and_intermediate_targets_both_games_seats(self):
        for game in ("tictactoe", "connect4"):
            session = self.session(game)
            with torch.no_grad():
                for parameter in session.model.parameters():
                    parameter.zero_()
                session.model.value.bias.fill_(float(np.arctanh(.8)))
            snapshot = FrozenModel(session.model)
            group = collect_episode_group(game, episode_count=2, master_seed=7, snapshot=snapshot,
                                           max_episode_steps=1)
            targets = make_targets(group, snapshot)
            self.assertEqual([e.seat for e in group.episodes], [0,1])
            self.assertTrue(all(e.steps[-1].transition.truncated for e in group.episodes))
            np.testing.assert_allclose(targets.returns, [.792,.792], atol=1e-6)
            np.testing.assert_array_equal(targets.intermediate_returns, targets.returns)
            np.testing.assert_allclose(targets.advantages, [-.008,-.008], atol=1e-6)
            with torch.no_grad():
                session.model.value.bias.zero_()
            np.testing.assert_array_equal(make_targets(group,snapshot).returns, targets.returns)
            with self.assertRaisesRegex(ValueError, "different collection weights"):
                make_targets(group,FrozenModel(session.model))

    def test_rule_termination_has_zero_bootstrap_and_terminal_mid_is_omitted(self):
        session = self.session()
        snapshot = FrozenModel(session.model)
        group = collect_episode_group("tictactoe", episode_count=2, master_seed=8, phase="teacher", payloads=("cells",))
        targets = make_targets(group, snapshot)
        expected = []
        mid_expected = []
        for episode in group.episodes:
            transitions = [s.transition for s in episode.steps]
            returns = discounted_returns([t.reward for t in transitions])
            expected.extend(returns)
            for transition, value in zip(transitions,returns):
                if transition.intermediate["info"]["next_player"] is not None:
                    mid_expected.append(value)
        np.testing.assert_array_equal(targets.returns, expected)
        np.testing.assert_array_equal(targets.intermediate_returns, mid_expected)
        self.assertLess(len(mid_expected), len(expected))

    def test_loss_formula_episode_mean_masks_and_detached_targets(self):
        session = self.session()
        with torch.no_grad():
            for parameter in session.model.parameters():
                parameter.zero_()
        snapshot = FrozenModel(session.model)
        group = collect_episode_group("tictactoe", episode_count=2, master_seed=4, snapshot=snapshot, max_episode_steps=1)
        targets = make_targets(group,snapshot)
        targets = replace(targets, returns=np.array([.5,-.5],dtype=np.float32),
                          advantages=np.array([.25,-.5],dtype=np.float32),
                          actor_weights=np.array([1,.99],dtype=np.float32),
                          intermediate_batch=None, intermediate_returns=np.empty(0,dtype=np.float32))
        terms = loss_terms(session.model,targets,phase="actor_critic",settings=session.settings)
        legal_counts = targets.batch.legal_mask.sum(axis=1)
        expected_actor = np.sum(targets.actor_weights * targets.advantages * np.log(legal_counts)) / 2
        self.assertAlmostEqual(float(terms["policy"].detach()), expected_actor, places=6)
        self.assertAlmostEqual(float(terms["decision_value"].detach()), .25)
        self.assertEqual(float(terms["intermediate_value"]), 0)
        self.assertAlmostEqual(float(terms["entropy"].detach()), np.log(legal_counts).mean(), places=6)
        teacher = loss_terms(session.model,targets,phase="teacher",settings=session.settings)
        self.assertAlmostEqual(float(teacher["policy"].detach()), np.log(legal_counts).mean(),places=6)
        self.assertAlmostEqual(float(teacher["total"].detach()), np.log(legal_counts).mean()+.125,places=6)
        terms["total"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in session.model.parameters()))

    def test_updates_teacher_and_actor_for_both_games_and_roundtrip_play(self):
        for game in ("tictactoe","connect4"):
            session = self.session(game)
            initial = FrozenModel(session.model).identity
            reports = list(session.run())
            self.assertEqual([r["episodes"] for r in reports], [2,1,2,2,1])
            self.assertEqual(session.completed, dict(teacher=3,actor_critic=5))
            self.assertEqual(session.next_episode_id,8)
            self.assertNotEqual(initial,FrozenModel(session.model).identity)
            self.assertTrue(all(r["before_identity"] != r["model_identity"] for r in reports))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"model.json"
                save_model(path,session.model,provenance=session.provenance())
                loaded=load_model(path,expected_contract=session.model.contract)
                for name,value in loaded.state_dict().items():
                    self.assertTrue(torch.equal(value,session.model.state_dict()[name]))
                for mode in ("policy","osla"):
                    result=evaluate_model(loaded,episodes=3,mode=mode,group_size=2)
                    self.assertEqual(result["combined"]["episodes"],3)
                    self.assertEqual([result["per_seat"][s]["episodes"] for s in (0,1)],[2,1])

    def test_checkpoint_resume_exact_weights_optimizer_history_and_rng(self):
        for game in ("tictactoe","connect4"):
            reference=self.session(game)
            list(reference.run())
            for split in (1,2,3): # teacher, phase boundary, actor-critic
                session=self.session(game)
                list(session.run(max_groups=split))
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/"training.pt"
                    save_checkpoint(path,session)
                    global_rng=torch.get_rng_state().clone()
                    resumed=load_checkpoint(path)
                    self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
                    list(resumed.run())
                self.assertEqual(reference.history,resumed.history)
                self.assertEqual(reference.provenance(),resumed.provenance())
                self.assertTrue(torch.equal(reference.torch_rng,resumed.torch_rng))
                for name,value in reference.model.state_dict().items():
                    self.assertTrue(torch.equal(value,resumed.model.state_dict()[name]))
                for p, q in zip(reference.optimizer.state.values(),resumed.optimizer.state.values()):
                    for name in p:
                        self.assertTrue(torch.equal(p[name],q[name]))

    def test_evaluation_does_not_change_training_weights_counters_or_rng(self):
        reference=self.session()
        session=self.session()
        reference.step()
        session.step()
        state=torch.get_rng_state().clone()
        for mode in ("policy","osla"):
            evaluate_model(session.model,episodes=3,mode=mode)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(reference.step(),session.step())

    def test_training_failure_blocks_checkpoint_and_settings_reject_invalid(self):
        session=self.session()
        with patch("tag_rust.learning.training.collect_episode_group",side_effect=RuntimeError("failed group")):
            with self.assertRaisesRegex(RuntimeError,"failed group"):
                session.step()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                save_checkpoint(Path(directory)/"failed.pt",session)
        for settings in (dict(group_size=0),dict(teacher_episodes=-1),dict(learning_rate=float("nan")),
                         dict(learner=True),dict(max_episode_steps=0),dict(threads=0)):
            with self.assertRaises(ValueError):
                TrainingSettings(**settings)

    def test_artifact_identity_shape_and_checkpoint_runtime_rejections(self):
        session=self.session()
        session.step()
        document=model_document(session.model)
        for mutate in (lambda d:d.update(version=99),
                       lambda d:d.update(model_identity="bad"),
                       lambda d:d["weights"]["value.bias"].update(shape=[100]),
                       lambda d:d["weights"]["value.bias"].update(values=[float("nan")]),
                       lambda d:d.update(hidden_sizes=[-1])):
            changed=deepcopy(document)
            mutate(changed)
            with self.assertRaises(ValueError):
                load_document(changed)
        with tag.create("connect4") as game:
            with self.assertRaisesRegex(ValueError,"contract differs"):
                load_document(document,expected_contract=ModelContract.from_spec(game.spec(),payloads=("cells",)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"training.pt"
            save_checkpoint(path,session)
            with self.assertRaisesRegex(ValueError,"evaluation-model JSON.*training checkpoints"):
                load_model(path)
            malformed=Path(directory)/"malformed.json"
            malformed.write_text("{unfinished",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"evaluation-model JSON"):
                load_model(malformed)
            original=torch.load(path,weights_only=True)
            for change in (lambda d:d["runtime"].update(torch="other"),
                           lambda d:d.update(next_episode_id=999),
                           lambda d:d["history"][0].update(first_episode_id=999),
                           lambda d:d["optimizer"]["param_groups"][0].update(lr=.5)):
                data=deepcopy(original)
                change(data)
                torch.save(data,path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)

    def test_failed_checkpoint_replacement_preserves_previous_file(self):
        session=self.session()
        session.step()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"training.pt"
            save_checkpoint(path,session)
            original=path.read_bytes()
            def fail_after_write(document,stream):
                stream.write(b"partial checkpoint")
                raise OSError("simulated write failure")
            with patch("tag_rust.learning.artifacts.torch.save",side_effect=fail_after_write):
                with self.assertRaisesRegex(OSError,"write failure"):
                    save_checkpoint(path,session)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(Path(directory).iterdir()),[path])
            self.assertEqual(load_checkpoint(path).history,session.history)


class PackageTopics(unittest.TestCase):
    def test_compatibility_imports_resolve_to_same_classes(self):
        from tag_rust.gym import GameEnv
        from tag_rust.episodes.gym import GameEnv as TopicEnv
        from tag_rust.api import EncodingIdentity
        from tag_rust.engine.api import EncodingIdentity as TopicIdentity
        self.assertIs(GameEnv,TopicEnv)
        self.assertIs(EncodingIdentity,TopicIdentity)
        if HAS_TORCH:
            from tag_rust.neural import PolicyValueNetwork
            from tag_rust.learning.network import PolicyValueNetwork as TopicNetwork
            self.assertIs(PolicyValueNetwork,TopicNetwork)
