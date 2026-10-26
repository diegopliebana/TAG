"""Self-play targets, exploration, bounded storage and complete CPU training cycles."""
from contextlib import ExitStack
from dataclasses import replace
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tag_rust as tag

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.learning.inputs import ModelContract,ValueSpec
    from tag_rust.learning.network import PolicyValueNetwork
    from tag_rust.learning.snapshot import FrozenModel
    from tag_rust.learning.artifacts import save_model,load_model
    from tag_rust.learning.selfplay import SelfPlaySettings,SelfPlaySession,save_checkpoint,load_checkpoint
    from tag_rust.learning.selfplay.settings import DataSettings,ExplorationSettings
    from tag_rust.learning.selfplay.collection import collect_games
    from tag_rust.learning.selfplay.data import PositionExample,ReplayBuffer,visit_target,owned_observation
    from tag_rust.learning.selfplay.loss import loss_terms
    from tag_rust.learning.selfplay.noise import RootNoiseEvaluator
    from tag_rust.learning.selfplay.seeds import derive_seed,generator


def configuration(game="tictactoe",**changes):
    value = {"schema_version":1,"game":game,"seed":42,"cycles":2,"games_per_cycle":2,"parallel_games":2,
        "search":{"schema_version":1,"tree_policy":"puct","rollout_length":0,"budgets":{"iterations":3},
                  "state_evaluator":{"name":"network_value"},"puct_preferences":{"source":"probabilities","evaluator":{"name":"network_policy"}}},
        "model":{"hidden_sizes":[4]},"optimization":{"updates_per_cycle":2,"batch_size":4},
        "evaluation":{"enabled":False}}
    value.update(changes)
    return value


def zero_model(contract):
    model = PolicyValueNetwork(contract,(4,))
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
    return model


def fixture(*,players=2,vector=False):
    with tag.create("tictactoe") as game:
        spec = game.spec()
        identity = replace(spec.identity,player_count=players)
        value = ValueSpec(players=tuple(reversed(range(players))) if vector else None,gamma=1.,bounds=None,
                          discount_clock="environment-action")
        original = ModelContract.from_spec(spec,payloads=("cells",),value=ValueSpec(gamma=1.))
        contract = replace(original,identity=identity,value=value)
        root = game.observe(0,("cells",))
        rows = []
        for observer in ((0,) if vector else range(players)):
            rows.append(replace(owned_observation(root),identity=identity,observer=observer,next_player=0,
                results=np.zeros(players,dtype=np.int8),action_mask=root.action_mask.copy() if observer==0 else np.zeros(9,dtype=np.bool_),
                offered_action_ids=root.offered_action_ids.copy() if observer==0 else np.array([],dtype=np.int64)))
        visits = np.array([2,1,0,0,0,0,0,0,0],dtype=np.int64)
        terminal = np.ones(players,dtype=np.float32)
        terminal[-1] = -1
        example = PositionExample(tuple(rows),0,(visits/3).astype(np.float32),visits,terminal,0,0,0,"fixture",42,9,{"settings":{"seed":42}})
        return contract,example


@unittest.skipUnless(HAS_TORCH,"optional PyTorch not installed")
class SelfPlayData(unittest.TestCase):
    def test_loss_arithmetic_uses_distributions_and_absolute_player_values(self):
        contract,example = fixture(players=3)
        model = zero_model(contract)
        policy,value = loss_terms(model,[example])
        self.assertAlmostEqual(float(policy.detach()),math.log(9),places=6)
        self.assertAlmostEqual(float(value.detach()),1.,places=6)
        # Consecutive decisions by the same player keep that explicit perspective.
        repeated = replace(example,ply=1)
        again_policy,again_value = loss_terms(model,[example,repeated])
        self.assertAlmostEqual(float(again_policy.detach()),math.log(9),places=6)
        self.assertAlmostEqual(float(again_value.detach()),1.,places=6)
        vector,example = fixture(players=3,vector=True)
        model = zero_model(vector)
        with torch.no_grad(): model.value.bias[:] = torch.tensor([.2,.4,.6])
        policy,value = loss_terms(model,[example])
        self.assertAlmostEqual(float(policy.detach()),math.log(9),places=6)
        expected = ((.2-(-1))**2+(.4-1)**2+(.6-1)**2)/3
        self.assertAlmostEqual(float(value.detach()),expected,places=6)

    def test_visit_targets_reject_fallback_illegal_ids_and_nonfinite_labels(self):
        contract,example = fixture()
        observation = example.observations[0]
        report = {"action_id":0,"root_actions":[{"action_id":i,"visits":0} for i in range(9)]}
        with self.assertRaisesRegex(ValueError,"completed root visit"): visit_target(observation,report)
        report["root_actions"][0]["visits"] = 3
        visits,policy = visit_target(observation,report)
        self.assertEqual(policy[0],1)
        report["root_actions"][1]["action_id"] = 99
        with self.assertRaises(ValueError): visit_target(observation,report)
        bad = replace(example,terminal_values=np.array([np.nan,1.],dtype=np.float32))
        with self.assertRaises(ValueError): loss_terms(zero_model(contract),[bad])

    def test_fifo_newest_group_byte_limits_and_sample_ownership(self):
        contract,example = fixture()
        positions = [replace(example,game_id=i) for i in range(4)]
        buffer = ReplayBuffer(DataSettings(capacity=2),contract)
        buffer.add_group(positions)
        self.assertEqual([p.game_id for p in buffer.positions],[2,3])
        self.assertEqual(buffer.dropped,2)
        fingerprint = buffer.digest()
        batch,indices = buffer.sample(8,generator(42))
        batch[0].policy[:] = 99
        self.assertEqual(buffer.digest(),fingerprint)
        _,again = buffer.sample(8,generator(42))
        self.assertEqual(indices,again)
        limited = ReplayBuffer(DataSettings(capacity=10,max_bytes=example.nbytes),contract)
        limited.add_group(positions)
        self.assertEqual(len(limited),1)
        with self.assertRaises(MemoryError):
            ReplayBuffer(DataSettings(max_bytes=example.nbytes-1),contract).add_group([example])
        newest = ReplayBuffer(DataSettings(mode="newest_group"),contract)
        newest.add_group(positions[:2]);newest.add_group(positions[2:])
        self.assertEqual([p.game_id for p in newest.positions],[2,3])

    def test_supervised_objective_decreases_on_a_fixed_batch(self):
        contract,example = fixture()
        model = zero_model(contract)
        optimizer = torch.optim.Adam(model.parameters(),lr=.05)
        def objective():
            policy,value = loss_terms(model,[example])
            return policy+value
        before = float(objective().detach())
        for _ in range(12):
            optimizer.zero_grad();loss=objective();loss.backward();optimizer.step()
        self.assertLess(float(objective().detach()),before)

    def test_root_noise_is_seeded_root_only_and_aligned_once_by_action_id(self):
        class Base:
            identity = "uniform-fixture"
            payloads = ()
            def evaluate(self,requests,*,max_bytes):
                return tuple(tag.MctsResponse.for_request(r,np.full(len(r.action_ids),1/len(r.action_ids))
                    if r.kind=="policy" else [.25,.75]) for r in requests)
        with tag.create("tictactoe") as game:
            request = tag.MctsRequest((1,1),"policy","policy","unused",0,0,(game.observe(0,()),),(0,1,2),0)
            noise = RootNoiseEvaluator(Base(),ExplorationSettings(noise_fraction=.5))
            request = replace(request,evaluator_identity=noise.identity)
            noise.bind(request,42)
            first, = noise.evaluate([request],max_bytes=10000)
            second = replace(request,id=(1,2),action_ids=(2,0,1))
            repeated, = noise.evaluate([second],max_bytes=10000)
            np.testing.assert_array_equal(repeated.values,first.values[[2,0,1]])
            deep = replace(request,id=(1,3),policy_node=7)
            unchanged, = noise.evaluate([deep],max_bytes=10000)
            np.testing.assert_allclose(unchanged.values,np.full(3,1/3))
            self.assertFalse(np.allclose(first.values,unchanged.values))
            self.assertAlmostEqual(first.values.sum(),1)
            noise.release([1]);self.assertFalse(noise._contexts)


@unittest.skipUnless(HAS_TORCH,"optional PyTorch not installed")
class SelfPlayCycles(unittest.TestCase):
    def test_collection_is_identical_with_different_independent_game_schedules(self):
        settings = SelfPlaySettings.from_dict(configuration(games_per_cycle=3))
        seed_session = SelfPlaySession(settings)
        snapshot = FrozenModel(zero_model(seed_session.model.contract))
        one = collect_games(replace(settings,parallel_games=1),snapshot)
        many = collect_games(replace(settings,parallel_games=3),snapshot)
        def decisions(result):
            return [(p.game_id,p.ply,p.actor,p.action,p.visits.tolist(),p.terminal_values.tolist(),p.search_seed,p.noise_seed)
                    for p in result.positions]
        self.assertEqual(decisions(one),decisions(many))
        self.assertEqual([game.game_id for game in many.games],[0,1,2])
        self.assertTrue(all(len(p.observations)==2 for p in many.positions))
        self.assertTrue(all(np.isin(p.terminal_values,[-1,0,1]).all() for p in many.positions))

    def test_both_games_train_export_reload_and_continue_from_latest_model(self):
        with tempfile.TemporaryDirectory() as folder:
            # Concrete placement-game training fixtures; basic registration alone
            # does not advertise self-play support for a new game.
            for game in ("tictactoe", "connect4"):
                session = SelfPlaySession(SelfPlaySettings.from_dict(configuration(game)))
                before = FrozenModel(session.model).identity
                reports = list(session.run())
                self.assertEqual(reports[0]["before_identity"],before)
                self.assertNotEqual(reports[0]["model_identity"],before)
                self.assertEqual(reports[1]["before_identity"],reports[0]["model_identity"])
                self.assertEqual(session.completed_cycles,2)
                path = Path(folder)/f"{game}.json"
                save_model(path,session.model,provenance=session.provenance())
                self.assertEqual(FrozenModel(load_model(path)).identity,reports[-1]["model_identity"])

    def test_checkpoint_resume_matches_weights_optimizer_samples_and_replay_order(self):
        settings = SelfPlaySettings.from_dict(configuration())
        uninterrupted = SelfPlaySession(settings)
        uninterrupted.run_cycle()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"training.pt"
            save_checkpoint(path,uninterrupted)
            resumed = load_checkpoint(path)
            first = uninterrupted.run_cycle();second = resumed.run_cycle()
            self.assertEqual(first["model_identity"],second["model_identity"])
            self.assertEqual(first["optimization"],second["optimization"])
            self.assertEqual(uninterrupted.buffer.digest(),resumed.buffer.digest())
            self.assertEqual(uninterrupted.sample_rng.bit_generator.state,resumed.sample_rng.bit_generator.state)
            for key,value in uninterrupted.model.state_dict().items():
                self.assertTrue(torch.equal(value,resumed.model.state_dict()[key]))
            for left,right in zip(uninterrupted.optimizer.state.values(),resumed.optimizer.state.values()):
                for key in left: self.assertTrue(torch.equal(left[key],right[key]))
            data = torch.load(path,weights_only=True)
            data["positions"][0]["terminal_values"][0] = float("nan")
            torch.save(data,path)
            with self.assertRaises(ValueError): load_checkpoint(path)

    def test_full_vector_model_newest_group_and_seat_balanced_evaluation(self):
        settings = SelfPlaySettings.from_dict(configuration(model={"hidden_sizes":[4],"value_output":"players","value_players":[1,0]},
            data={"mode":"newest_group"},evaluation={"enabled":True,"games":2,
                "opponent_search":{"schema_version":1,"budgets":{"iterations":2},"rollout_length":0}}))
        session = SelfPlaySession(settings)
        reports = list(session.run())
        self.assertEqual(session.model.contract.value.players,(1,0))
        self.assertTrue(all(len(p.observations)==1 and p.game_id>=2 for p in session.buffer.positions))
        for report in reports:
            self.assertEqual(report["evaluation"]["completed_games"],2)
            self.assertEqual([row["games"] for row in report["evaluation"]["per_seat"]],[1,1])
            self.assertFalse(report["evaluation"]["root_noise"])

    def test_failed_collection_blocks_checkpoint_and_preserves_prior_file(self):
        settings = SelfPlaySettings.from_dict(configuration(limits={"max_moves":1}))
        session = SelfPlaySession(settings)
        initial = FrozenModel(session.model).identity
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"before.pt"
            save_checkpoint(path,session);original=path.read_bytes()
            with self.assertRaisesRegex(RuntimeError,"max_moves"): session.run_cycle()
            self.assertEqual(FrozenModel(session.model).identity,initial)
            self.assertEqual(len(session.buffer),0)
            with self.assertRaises(RuntimeError): save_checkpoint(path,session)
            self.assertEqual(path.read_bytes(),original)

    def test_optimizer_failure_and_collection_limits_cannot_replace_a_checkpoint(self):
        session = SelfPlaySession(SelfPlaySettings.from_dict(configuration()))
        session.run_cycle()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"completed.pt"
            save_checkpoint(path,session);original=path.read_bytes()
            with patch.object(session.optimizer,"step",side_effect=RuntimeError("update failed")):
                with self.assertRaisesRegex(RuntimeError,"update failed"): session.run_cycle()
            with self.assertRaises(RuntimeError): save_checkpoint(path,session)
            self.assertEqual(path.read_bytes(),original)
        for limits in ({"max_collection_positions":1},{"max_collection_bytes":1}):
            limited = SelfPlaySession(SelfPlaySettings.from_dict(configuration(limits=limits)))
            with self.assertRaises(MemoryError): limited.run_cycle()
            self.assertEqual(len(limited.buffer),0)

    def test_evaluation_preserves_training_weights_data_and_random_streams(self):
        from tag_rust.learning.selfplay.evaluation import evaluate
        session = SelfPlaySession(SelfPlaySettings.from_dict(configuration()))
        session.run_cycle()
        fingerprint = session.buffer.digest()
        rng = copy.deepcopy(session.sample_rng.bit_generator.state)
        torch_rng = session.torch_rng.clone()
        snapshot = FrozenModel(session.model)
        evaluation = replace(session.settings.evaluation,enabled=True,games=2,
                             opponent_search={"schema_version":1,"budgets":{"iterations":2},"rollout_length":0})
        report = evaluate(replace(session.settings,evaluation=evaluation),snapshot,0)
        self.assertEqual(report["completed_games"],2)
        self.assertEqual(session.buffer.digest(),fingerprint)
        self.assertEqual(session.sample_rng.bit_generator.state,rng)
        self.assertTrue(torch.equal(session.torch_rng,torch_rng))
        self.assertEqual(FrozenModel(session.model).identity,snapshot.identity)

    def test_configuration_paths_limits_and_model_semantics_are_explicit(self):
        for update in ({"unknown":3},{"exploration":{"temperature":0}},
                       {"data":{"capacity":0}},{"optimization":{"value_weight":0,"policy_weight":0}}):
            with self.assertRaises(ValueError): SelfPlaySettings.from_dict(configuration(**update))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/"search.json").write_text('{"schema_version":1}')
            (folder/"config.json").write_text('{"schema_version":1,"search_config_file":"search.json","exploration":{"root_noise":false},"output":{"model_file":"model.json"}}')
            settings = SelfPlaySettings.load(folder/"config.json")
            self.assertEqual(settings.output.model_file,str(folder/"model.json"))
            # PowerShell can prefix both the main and referenced JSON with a BOM.
            for name in ("config.json", "search.json"):
                path = folder/name
                path.write_text(path.read_text(encoding="utf-8"),encoding="utf-8-sig")
            self.assertEqual(SelfPlaySettings.load(folder/"config.json"),settings)
        with tag.create("tictactoe") as game:
            old_model = PolicyValueNetwork(ModelContract.from_spec(game.spec(),payloads=("cells",)),(4,))
            with self.assertRaisesRegex(ValueError,"contract"):
                SelfPlaySession(SelfPlaySettings.from_dict(configuration()),initial_model=old_model)

    def test_automatic_seed_and_incompatible_checkpoint_state_are_explicit(self):
        with patch("tag_rust.learning.selfplay.training._engine.automatic_search_seed",return_value=123) as source:
            session = SelfPlaySession(SelfPlaySettings.from_dict(configuration(seed=None)))
            self.assertEqual(session.settings.seed,123)
            source.assert_called_once()
        session.run_cycle()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"state.pt"
            save_checkpoint(path,session)
            original = torch.load(path,weights_only=True)
            for field in ("version","runtime","updates","positions","sample_rng"):
                document = copy.deepcopy(original)
                if field == "version": document["version"] = 999
                elif field == "runtime": document["runtime"]["torch"] = "different-runtime"
                elif field == "updates": document["updates"] += 1
                elif field == "positions": document["positions"].append(document["positions"][0])
                else: document["sample_rng"]["bit_generator"] = "MT19937"
                torch.save(document,path)
                with self.assertRaises(ValueError): load_checkpoint(path)
            with self.assertRaises(ValueError): load_checkpoint(path,max_bytes=1)


if __name__ == "__main__":
    unittest.main()
