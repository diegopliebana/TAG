"""Completed stochastic training games, seed provenance and exact cycle resumes."""
from dataclasses import replace
from contextlib import redirect_stdout,redirect_stderr
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tag_rust as tag
from test_cantstop import same_view
from test_selfplay import configuration

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.learning.artifacts import save_model,load_model
    from tag_rust.learning.snapshot import FrozenModel
    from tag_rust.learning.selfplay import SelfPlaySettings,SelfPlaySession,save_checkpoint,load_checkpoint
    from tag_rust.learning.selfplay.collection import collect_games
    from tag_rust.learning.selfplay.seeds import derive_seed,workflow_seed,game_parameters


def stochastic_settings(players=3, **changes):
    value = configuration("cantstop", players=players,
        parameters={"rules":{"column_heights":[2]*11,"columns_to_win":1}},
        model={"payloads":["permanent","temporary","owners","dice","context"],
               "hidden_sizes":[4],"value_output":"players"}, limits={"max_moves":1000})
    value.update(changes)
    return SelfPlaySettings.from_dict(value)


@unittest.skipUnless(HAS_TORCH,"optional PyTorch not installed")
class CantStopSelfPlay(unittest.TestCase):
    def test_versioned_seed_fixtures_and_unchanged_placement_recipe(self):
        settings = stochastic_settings()
        self.assertEqual(workflow_seed(settings,"selfplay",purpose="game-chance"),16813839899447826188)
        self.assertEqual(workflow_seed(settings,"selfplay",purpose="search"),13558718246091080344)
        self.assertEqual(workflow_seed(settings,"selfplay",purpose="root-noise"),4033141263566526617)
        self.assertEqual(workflow_seed(settings,"initialization",purpose="model"),13752979996328685687)
        self.assertEqual(workflow_seed(settings,"training",purpose="minibatches"),6472141716372111141)
        self.assertEqual(derive_seed(42,"selfplay"),16396464405638008509)
        placement = SelfPlaySettings.from_dict(configuration())
        self.assertEqual(workflow_seed(placement,"selfplay"),derive_seed(42,"selfplay"))
        seeds = [game_parameters(settings,index)["seed"] for index in range(4)]
        self.assertEqual(len(set(seeds)),4)
        self.assertNotIn("seed", settings.parameters)

    def test_collection_replays_actual_dice_and_is_independent_of_game_schedule(self):
        for players in (2,3,4):
            with self.subTest(players=players):
                settings = stochastic_settings(players)
                session = SelfPlaySession(settings)
                # Constant predictions remove floating-point batching variation.
                with torch.no_grad():
                    for parameter in session.model.parameters(): parameter.zero_()
                snapshot = FrozenModel(session.model)
                single = collect_games(replace(settings,parallel_games=1),snapshot)
                grouped = collect_games(replace(settings,parallel_games=2),snapshot)
                def trace(result):
                    return [(g.game_id,g.game_seed,g.outcomes,
                        [(p.actor,p.action,p.visits.tolist(),p.terminal_values.tolist(),p.search_seed,p.noise_seed)
                         for p in g.positions]) for g in result.games]
                self.assertEqual(trace(single),trace(grouped))
                self.assertEqual(len({g.game_seed for g in grouped.games}),2)
                terminal_limit = replace(settings.limits,max_moves=len(single.games[0].positions))
                bounded = collect_games(replace(settings,games_per_cycle=1,parallel_games=1,limits=terminal_limit),snapshot)
                self.assertEqual(trace(bounded),trace(single)[:1])
                repeated = False
                for result in grouped.games:
                    self.assertEqual(result.game_seed,game_parameters(settings,result.game_id)["seed"])
                    with tag.create("cantstop",game_parameters(settings,result.game_id),players=players) as game:
                        previous = None
                        for position in result.positions:
                            self.assertEqual(len(position.observations),1)
                            observation = position.observations[0]
                            same_view(self,observation,game.observe(game.next_player))
                            self.assertEqual(position.actor,game.next_player)
                            repeated |= previous == position.actor
                            previous = position.actor
                            self.assertEqual(float(position.policy.sum()),1.)
                            self.assertFalse(position.policy[~observation.action_mask].any())
                            self.assertEqual(position.search_seed,workflow_seed(settings,"selfplay",result.game_id,position.ply,"search"))
                            game.apply(position.action,position=game.position)
                        self.assertIsNone(game.next_player)
                        self.assertEqual(tuple(game.observe(0,()).results),result.outcomes)
                    expected = np.array([1. if code==1 else -1. for code in result.outcomes],dtype=np.float32)
                    for position in result.positions:
                        np.testing.assert_array_equal(position.terminal_values,expected)
                self.assertTrue(repeated)

    def test_all_player_counts_train_export_and_resume_both_buffer_modes_exactly(self):
        for players in (2,3,4):
            for mode in ("replay_buffer","newest_group"):
                with self.subTest(players=players,mode=mode),tempfile.TemporaryDirectory() as folder:
                    settings = stochastic_settings(players,data={"mode":mode})
                    session = SelfPlaySession(settings)
                    initial = session.initial_identity
                    first = session.run_cycle()
                    self.assertNotEqual(first["model_identity"],initial)
                    self.assertEqual(session.model.contract.value.players,tuple(range(players)))
                    self.assertEqual(first["evaluation"],{"enabled":False})
                    self.assertEqual([row["games"] for row in first["per_seat"]],[2]*players)
                    self.assertEqual(first["combined"]["games"],2*players)
                    self.assertEqual(first["combined"]["wins"],2)
                    self.assertEqual(first["combined"]["losses"],2*(players-1))
                    path = Path(folder)/"training.pt"
                    save_checkpoint(path,session)
                    self.assertEqual(torch.load(path,weights_only=True)["version"],2)
                    with patch("tag_rust.learning.selfplay.training._engine.automatic_search_seed",side_effect=AssertionError("resume reseeded")):
                        resumed = load_checkpoint(path)
                    next_report = session.run_cycle()
                    resumed_report = resumed.run_cycle()
                    for field in ("before_identity","model_identity","games","optimization","buffer_digest","per_seat","combined"):
                        self.assertEqual(next_report[field],resumed_report[field])
                    self.assertEqual(session.buffer.digest(),resumed.buffer.digest())
                    self.assertEqual(session.sample_rng.bit_generator.state,resumed.sample_rng.bit_generator.state)
                    self.assertTrue(torch.equal(session.torch_rng,resumed.torch_rng))
                    for key,tensor in session.model.state_dict().items():
                        self.assertTrue(torch.equal(tensor,resumed.model.state_dict()[key]))
                    for left,right in zip(session.optimizer.state.values(),resumed.optimizer.state.values()):
                        for key in left: self.assertTrue(torch.equal(left[key],right[key]))
                    if mode=="newest_group":
                        self.assertTrue(all(p.game_id>=2 for p in resumed.buffer.positions))
                    exported = Path(folder)/"model.json"
                    save_model(exported,resumed.model,provenance=resumed.provenance())
                    loaded = load_model(exported)
                    self.assertEqual(FrozenModel(loaded).identity,resumed_report["model_identity"])
                    other = SelfPlaySession(settings,initial_model=loaded)
                    self.assertEqual(other.initial_identity,resumed_report["model_identity"])
                    from tag_rust.search.neural_mcts import NeuralMctsEvaluator
                    with tag.create("cantstop",game_parameters(settings,100),players=players) as game:
                        game.apply(0,position=game.position)
                        evaluator = NeuralMctsEvaluator(FrozenModel(loaded),turn_context="own_turn")
                        search_settings = dict(settings.search,seed=101)
                        with tag.MctsSearch(game,search_settings,bindings={
                                "network_value":evaluator,"network_policy":evaluator}) as search:
                            with tag.MctsBatch([search]) as batch:
                                report, = batch.run()
                                self.assertFalse(batch.errors)
                            game.apply(report["action_id"],position=search.position)

    def test_automatic_master_seed_resolves_once_and_repeats_explicit_run(self):
        with patch("tag_rust.learning.selfplay.training._engine.automatic_search_seed",return_value=77) as source:
            automatic = SelfPlaySession(stochastic_settings(seed=None))
            source.assert_called_once()
        explicit = SelfPlaySession(stochastic_settings(seed=77))
        left,right = automatic.run_cycle(),explicit.run_cycle()
        for field in ("model_identity","games","optimization","buffer_digest"):
            self.assertEqual(left[field],right[field])
        self.assertEqual(automatic.provenance()["seed_derivation_version"],2)

    def test_incomplete_collection_and_update_failure_preserve_last_checkpoint(self):
        for limits in ({"max_moves":1},{"max_collection_positions":1},{"max_collection_bytes":1}):
            with self.subTest(limits=limits),tempfile.TemporaryDirectory() as folder:
                session = SelfPlaySession(stochastic_settings(limits=limits))
                path = Path(folder)/"last.pt"
                save_checkpoint(path,session)
                saved = path.read_bytes()
                with self.assertRaises((RuntimeError,MemoryError)): session.run_cycle()
                self.assertEqual(len(session.buffer),0)
                self.assertEqual(session.updates,0)
                self.assertEqual(FrozenModel(session.model).identity,session.initial_identity)
                with self.assertRaises(RuntimeError): save_checkpoint(path,session)
                self.assertEqual(path.read_bytes(),saved)
                self.assertEqual(load_checkpoint(path).completed_cycles,0)
        # A completed peer is also excluded when a later game in the same group fails.
        session = SelfPlaySession(stochastic_settings(parallel_games=1))
        real_create = tag.create
        created = []
        def fail_second(*args,**kwargs):
            if created:
                raise RuntimeError("second game failed")
            game = real_create(*args,**kwargs)
            created.append(game)
            return game
        with patch("tag_rust.learning.selfplay.collection.create",side_effect=fail_second) as factory:
            with self.assertRaisesRegex(RuntimeError,"second game failed"): session.run_cycle()
            self.assertEqual(factory.call_count,2)
        self.assertEqual(len(session.buffer),0)
        self.assertEqual(session.updates,0)
        self.assertEqual(FrozenModel(session.model).identity,session.initial_identity)
        session = SelfPlaySession(stochastic_settings())
        session.run_cycle()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"last.pt"
            save_checkpoint(path,session)
            saved = path.read_bytes()
            with patch.object(session.optimizer,"step",side_effect=RuntimeError("failed update")):
                with self.assertRaises(RuntimeError): session.run_cycle()
            with self.assertRaises(RuntimeError): save_checkpoint(path,session)
            self.assertEqual(path.read_bytes(),saved)
            self.assertEqual(load_checkpoint(path).completed_cycles,1)

    def test_contract_and_checkpoint_provenance_rejections(self):
        for changes in ({"parameters":{"seed":42}}, {"model":{"value_output":"observer"}},
                        {"model":{"value_output":"players","value_players":[2,1,0]}}):
            with self.assertRaises(ValueError): stochastic_settings(**changes)
        session = SelfPlaySession(stochastic_settings())
        session.run_cycle()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"state.pt"
            save_checkpoint(path,session)
            saved = torch.load(path,weights_only=True)
            def wrong_seed(data): data["history"][0]["games"][0]["game_seed"] += 1
            def wrong_format(data): data["version"] = 1
            def wrong_recipe(data): data["model"]["provenance"]["seed_derivation_version"] = 1
            def wrong_rules(data): data["settings"]["parameters"]["rules"]["markers"] = 2
            def wrong_players(data): data["settings"]["players"] = 4
            def wrong_search(data): data["positions"][0]["search_seed"] += 1
            def wrong_summary(data): data["history"][0]["combined"]["wins"] += 1
            for mutate in (wrong_seed,wrong_format,wrong_recipe,wrong_rules,wrong_players,wrong_search,wrong_summary):
                altered = copy.deepcopy(saved)
                mutate(altered)
                torch.save(altered,path)
                with self.assertRaises(ValueError): load_checkpoint(path)
            model_path = Path(folder)/"model.json"
            save_model(model_path,session.model)
            model = load_model(model_path)
            for settings in (stochastic_settings(2),stochastic_settings(parameters={"rules":{"markers":2}})):
                with self.assertRaises(ValueError): SelfPlaySession(settings,initial_model=model)

    def test_cli_exposes_automatic_seed_before_a_failed_collection(self):
        path = Path(__file__).resolve().parents[1]/"examples/selfplay_train.py"
        spec = importlib.util.spec_from_file_location("selfplay_train_cli",path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder)/"config.json"
            config.write_text(json.dumps(stochastic_settings(seed=None,limits={"max_moves":1}).document()))
            output,errors = io.StringIO(),io.StringIO()
            with patch("sys.argv",[str(path),"--config",str(config)]),redirect_stdout(output),redirect_stderr(errors):
                with patch("tag_rust.learning.selfplay.training._engine.automatic_search_seed",return_value=123):
                    with self.assertRaises(SystemExit) as failure: module.main()
            self.assertEqual(failure.exception.code,2)
            self.assertEqual(json.loads(output.getvalue()),{
                "master_seed":123,"seed_derivation_version":2,"completed_cycles":0})
            self.assertIn("max_moves",errors.getvalue())


if __name__ == "__main__":
    unittest.main()
