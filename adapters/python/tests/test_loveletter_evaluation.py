"""Frozen Love Letter evaluation, public reports and native RNG continuation."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import tag_rust as tag
from tag_rust import _engine
from test_multiplayer_evaluation import HAS_TORCH, ADAPTER
from test_loveletter_selfplay import letter_settings

if HAS_TORCH:
    import torch
    from tag_rust.learning.selfplay import SelfPlaySession, evaluate_model, save_checkpoint, load_checkpoint
    from tag_rust.learning.selfplay.seeds import game_parameters
    from tag_rust.learning.snapshot import FrozenModel
    from tag_rust.learning.artifacts import save_model


class LoveLetterNativeContinuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(["cargo", "run", "--offline", "--quiet", "--manifest-path",
            str(ADAPTER / "Cargo.toml"), "--example", "loveletter_opponents_reference"],
            text=True, capture_output=True, check=True)
        cls.records = json.loads(result.stdout)

    def test_all_native_types_match_direct_agents_at_every_seat_with_chunked_advancement(self):
        for case in self.records:
            with self.subTest(players=case["players"], seat=case["candidate"], kind=case["kind"]):
                with tag.create("loveletter", case["parameters"], players=case["players"]) as game:
                    agents = []
                    for seat in range(case["players"]):
                        agent = {"type": case["kind"], "seed": 100 + seat}
                        if case["kind"] == "osla": agent["repetitions_per_action"] = 2
                        if case["kind"] == "mcts":
                            agent["config"] = {"schema_version": 1, "rollout_length": 0, "budgets": {"iterations": 5}}
                        agents.append(None if seat == case["candidate"] else agent)
                    native = _engine.NativeOpponents(game._native, case["candidate"], json.dumps(agents))
                    actions = []
                    while True:
                        progress = json.loads(native.advance(game._native, 1, True))
                        actions.extend(progress["actions"])
                        self.assertNotEqual(progress["reason"], "failure", progress)
                        if progress["reason"] == "finished": break
                        if game.next_player == case["candidate"]:
                            view = game.observe(game.next_player, ())
                            actions.append(json.loads(native.apply_candidate(game._native,
                                int(view.offered_action_ids[0]), view.position)))
                    self.assertEqual(actions, case["actions"])
                    self.assertEqual(json.loads(native.state(game._native)),
                                     {k: case[k] for k in ("scores", "outcomes")})
                    self.assertTrue(all(set(a) == {"actor", "action_id"} for a in actions))

    def test_native_lineup_rejects_foreign_stale_and_invalid_candidate_actions_atomically(self):
        with tag.create("loveletter", {"seed": 42}) as game, game.clone() as clone:
            native = _engine.NativeOpponents(game._native, 0, json.dumps([None, {"type": "random", "seed": 1}]))
            before = game.position
            with self.assertRaises(ValueError): native.apply_candidate(game._native, 52, before)
            self.assertEqual(game.position, before)
            with self.assertRaises(tag.StalePositionError): native.advance(clone._native)
            view = game.observe(0, ())
            native.apply_candidate(game._native, int(view.offered_action_ids[0]), before)
            with self.assertRaises(tag.StalePositionError): native.advance(clone._native)
            game.reset()
            with self.assertRaises(tag.StalePositionError): native.advance(game._native)


@unittest.skipUnless(HAS_TORCH, "optional PyTorch not installed")
class LoveLetterEvaluation(unittest.TestCase):
    def settings(self, players=3, **changes):
        evaluation = {"enabled": True, "games": players, "opponent": {"name": "random", "agent": {"type": "random"}},
                      "ranking": "score", "include_actions": True, "max_actions": 4096}
        evaluation.update(changes)
        return letter_settings(players, evaluation=evaluation)

    def test_complete_reproducible_games_all_counts_replay_and_keep_training_unchanged(self):
        for n in (2, 3, 4):
            settings = self.settings(n)
            session = SelfPlaySession(settings)
            weights = copy.deepcopy(session.model.state_dict())
            rng = copy.deepcopy(session.sample_rng.bit_generator.state)
            torch_rng = session.torch_rng.clone()
            snapshot = FrozenModel(session.model)
            report = evaluate_model(settings, snapshot)
            self.assertEqual(report["information_access"], "privileged-research")
            self.assertEqual(report["completed_games"], n, report["failures"])
            self.assertEqual([r["games"] for r in report["per_seat"]], [1]*n)
            self.assertEqual(report["games"], evaluate_model(settings, snapshot)["games"])
            for row in report["games"]:
                self.assertEqual(row["first_player"], 0)
                with tag.create("loveletter", game_parameters(settings, row["game_id"], phase="evaluation"), players=n) as game:
                    for event in row["actions"]:
                        self.assertEqual(game.next_player, event["actor"])
                        game.apply(event["action_id"], position=game.position)
                    self.assertIsNone(game.next_player)
                    view = game.observe(0, ("players",))
                    self.assertEqual(view.results.tolist(), row["outcomes"])
                    scores = [int(p[2])*65536+int(p[3]) for p in view.payloads["players"]]
                    self.assertEqual(scores, row["scores"])
                self.assertEqual(row["moves"], len(row["actions"]))
            self.assertTrue(all(torch.equal(v, session.model.state_dict()[k]) for k,v in weights.items()))
            self.assertEqual(rng, session.sample_rng.bit_generator.state)
            self.assertTrue(torch.equal(torch_rng, session.torch_rng))
            self.assertEqual(len(session.buffer), 0)

    def test_rotation_grid_fixed_overrides_mixed_and_default_opponents(self):
        settings = self.settings(games=9, max_actions=1)
        session = SelfPlaySession(settings)
        snapshot = FrozenModel(session.model)
        rows = evaluate_model(settings, snapshot)["games"]
        self.assertEqual({(r["candidate_seat"], r["first_player"]) for r in rows},
                         {(s,f) for s in range(3) for f in range(3)})
        fixed = replace(settings, parameters=dict(settings.parameters, first_player=2),
                        evaluation=replace(settings.evaluation, positions="fixed", candidate_seat=1, games=2))
        self.assertEqual([(r["candidate_seat"], r["first_player"]) for r in evaluate_model(fixed,snapshot)["games"]], [(1,2)]*2)
        entries = [{"name":"osla", "agent":{"type":"osla", "repetitions_per_action":2}},
                   {"name":"mcts", "agent":{"type":"mcts", "config":{"schema_version":1,"rollout_length":0,"budgets":{"iterations":3}}}}]
        mixed = replace(settings, evaluation=replace(settings.evaluation, opponent=None, opponents=entries, games=1, max_actions=4096))
        self.assertEqual(evaluate_model(mixed,snapshot)["completed_games"],1)
        default = replace(mixed, evaluation=replace(mixed.evaluation, opponents=None,
            opponent_search={"schema_version":1,"rollout_length":0,"budgets":{"iterations":2}}))
        self.assertEqual(evaluate_model(default,snapshot)["completed_games"],1)
        resumed = evaluate_model(settings,snapshot,cycle=1)["games"]
        self.assertEqual(resumed[0]["game_id"],9)
        self.assertEqual(resumed[0]["first_player"],0)

    def test_auto_seed_limits_shared_wins_and_failure_preserves_peers(self):
        settings = self.settings(games=9)
        session = SelfPlaySession(settings)
        snapshot = FrozenModel(session.model)
        with patch("tag_rust._engine.automatic_search_seed", return_value=123) as source:
            auto = evaluate_model(replace(settings,seed=None),snapshot)
            source.assert_called_once()
        explicit = evaluate_model(replace(settings,seed=123),snapshot)
        self.assertEqual(auto["games"],explicit["games"])
        limited = evaluate_model(replace(settings,evaluation=replace(settings.evaluation,max_actions=1)),snapshot)
        self.assertEqual(limited["truncated_games"],9)
        self.assertIsNone(limited["combined"]["win_rate"])
        self.assertTrue(all(r["outcomes"] is None and "finishing_positions" not in r for r in limited["games"]))
        from tag_rust.learning.selfplay import lineup_evaluation as workflow
        original = workflow.start_search
        def fail_second(game,settings,configuration,evaluator):
            if game.seed == explicit["games"][1]["game_seed"]: raise RuntimeError("inference failed")
            return original(game,settings,configuration,evaluator)
        with patch.object(workflow,"start_search",side_effect=fail_second):
            failed=evaluate_model(replace(settings,seed=123),snapshot)
        self.assertEqual(failed["completed_games"],1)
        self.assertEqual(failed["failed_games"],1)
        self.assertEqual(failed["games"][0],explicit["games"][0])
        full = explicit["games"][0]
        exact = replace(settings,seed=123,evaluation=replace(settings.evaluation,games=1,max_actions=full["moves"]))
        self.assertEqual(evaluate_model(exact,snapshot)["completed_games"],1)
        # Fixed real game: seats 1 and 2 share the winning token.
        four = self.settings(4,games=1)
        four = replace(four,parameters={"rules":{"winning_tokens":[1,1,1]}})
        four_session=SelfPlaySession(four)
        shared=evaluate_model(replace(four,seed=28),FrozenModel(four_session.model))
        winners=[r for r in shared["games"] if r["outcomes"].count(1)>1]
        self.assertTrue(winners)
        for row in winners:
            self.assertNotIn(2,row["outcomes"])
            for seat,code in enumerate(row["outcomes"]):
                if code==1: self.assertEqual(row["finishing_positions"][seat],1.5)

    def test_cycle_checkpoint_continuation_and_cli_export(self):
        settings=self.settings(games=3,max_actions=1)
        session=SelfPlaySession(settings)
        first=session.run_cycle()
        self.assertEqual(first["evaluation"]["truncated_games"],3)
        baseline=SelfPlaySession(replace(settings,evaluation=replace(settings.evaluation,enabled=False)))
        without_evaluation=baseline.run_cycle()
        self.assertEqual(first["model_identity"],without_evaluation["model_identity"])
        self.assertEqual(first["optimization"],without_evaluation["optimization"])
        self.assertEqual(session.sample_rng.bit_generator.state,baseline.sample_rng.bit_generator.state)
        self.assertTrue(torch.equal(session.torch_rng,baseline.torch_rng))
        self.assertEqual(len(session.buffer),len(baseline.buffer))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            checkpoint=root/"state.pt"
            save_checkpoint(checkpoint,session)
            resumed=load_checkpoint(checkpoint)
            left,right=session.run_cycle(),resumed.run_cycle()
            self.assertEqual(left["optimization"],right["optimization"])
            self.assertEqual(left["evaluation"]["games"],right["evaluation"]["games"])
            self.assertEqual(left["model_identity"],right["model_identity"])
            model=root/"model.json"
            save_model(model,session.model)
            document={"schema_version":1,"seed":42,"parameters":settings.parameters,"search":settings.search,
                      "evaluation":{"games":1,"opponent":{"name":"random","agent":{"type":"random"}}},
                      "output":{"report_file":"report.json"}}
            config=root/"eval.json"
            config.write_text(json.dumps(document),encoding="utf-8")
            command=[sys.executable,str(ADAPTER/'examples/selfplay_evaluate.py'),
                     '--model',str(model),'--config',str(config)]
            result=subprocess.run(command,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=root/"report.json"
            saved=json.loads(report.read_text())
            self.assertEqual(saved["evaluation"]["completed_games"],1)
            before=report.read_bytes()
            document["players"]=4
            config.write_text(json.dumps(document),encoding="utf-8")
            result=subprocess.run(command,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(report.read_bytes(),before)

    def test_configuration_rejects_invalid_lineups_and_incompatible_models(self):
        for change in ({"opponents":[],"opponent":None},
                       {"opponents":[{"name":"r","agent":{"type":"random"}}],"opponent":None},
                       {"max_actions":0}, {"ranking":"winner"}):
            with self.assertRaises(ValueError): self.settings(**change)
        invalid=self.settings(opponent={"name":"bad","agent":{"type":"mcts","config":{
            "schema_version":1,"state_evaluator":{"name":"missing"}}}})
        with self.assertRaises(ValueError): SelfPlaySession(invalid)

    def test_evaluation_search_inputs_do_not_depend_on_actual_hidden_world(self):
        from contextlib import ExitStack
        from test_loveletter_mcts import paired_worlds
        from test_loveletter import same_view
        from tag_rust.learning.selfplay.collection import neural_evaluator, move_settings, start_search
        settings=replace(self.settings(), parameters={})
        session=SelfPlaySession(settings)
        snapshot=FrozenModel(session.model)
        with ExitStack() as stack:
            worlds=paired_worlds(stack)
            requests=[]
            reports=[]
            for game in worlds:
                evaluator=neural_evaluator(snapshot)
                saved=[]
                original=evaluator.evaluate
                def record(rows, *, max_bytes, original=original, saved=saved):
                    saved.extend(rows)
                    return original(rows,max_bytes=max_bytes)
                with patch.object(evaluator,"evaluate",side_effect=record):
                    with start_search(game,settings,move_settings(settings,0,0,phase="evaluation",evaluation=True),evaluator) as search:
                        with tag.MctsBatch([search]) as batch:
                            report,=batch.run()
                            self.assertFalse(batch.errors)
                reports.append(report)
                requests.append(saved)
            self.assertEqual(reports[0]["action_id"],reports[1]["action_id"])
            self.assertEqual(len(requests[0]),len(requests[1]))
            self.assertTrue(requests[0])
            for left,right in zip(*requests):
                self.assertEqual(left.rows,1)
                self.assertEqual(left.deciding_player,right.deciding_player)
                a,b=left.observations[0],right.observations[0]
                self.assertEqual(a.observer,left.deciding_player)
                self.assertNotIn("last_private_history",a.payloads)
                same_view(self,a,b)
