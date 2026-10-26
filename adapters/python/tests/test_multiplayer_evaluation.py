"""Native opponent continuation, candidate seats and frozen-model evaluation reports."""
from contextlib import ExitStack
from dataclasses import replace
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tag_rust as tag
from tag_rust import _engine

ADAPTER = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tag_rust.learning.selfplay import SelfPlaySession,SelfPlaySettings,evaluate_model,save_checkpoint,load_checkpoint
    from tag_rust.learning.selfplay.lineup_evaluation import finishing_positions,summarize
    from tag_rust.learning.snapshot import FrozenModel
    from test_cantstop_selfplay import stochastic_settings


class NativeOpponentContinuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(["cargo","run","--offline","--quiet","--manifest-path",str(ADAPTER/"Cargo.toml"),
            "--example","cantstop_opponents_reference"],text=True,capture_output=True,check=True)
        cls.records = json.loads(result.stdout)

    def test_all_native_types_match_direct_agents_at_every_seat_with_chunked_advancement(self):
        for case in self.records:
            with self.subTest(players=case["players"],seat=case["candidate"],kind=case["kind"]):
                with tag.create("cantstop",case["parameters"],players=case["players"]) as game:
                    agents = []
                    for seat in range(case["players"]):
                        agent = {"type":case["kind"],"seed":100+seat}
                        if case["kind"]=="osla": agent["repetitions_per_action"]=2
                        if case["kind"]=="mcts": agent["config"]={"schema_version":1,"rollout_length":0,"budgets":{"iterations":5}}
                        agents.append(None if seat==case["candidate"] else agent)
                    native = _engine.NativeOpponents(game._native,case["candidate"],json.dumps(agents))
                    actions = []
                    while True:
                        # One native action per call forces checkpoint restoration,
                        # including between repeated decisions by the same opponent.
                        progress = json.loads(native.advance(game._native,1,True))
                        actions.extend(progress["actions"])
                        self.assertNotEqual(progress["reason"],"failure")
                        if progress["reason"]=="finished": break
                        if game.next_player==case["candidate"]:
                            view = game.observe(game.next_player,())
                            actions.append(json.loads(native.apply_candidate(game._native,int(view.offered_action_ids[0]),view.position)))
                    self.assertEqual(actions,case["actions"])
                    self.assertEqual(json.loads(native.state(game._native)),{k:case[k] for k in ("scores","outcomes")})

    def test_native_lineup_rejects_foreign_stale_and_invalid_candidate_actions_atomically(self):
        with ExitStack() as stack:
            game = stack.enter_context(tag.create("cantstop",{"seed":42}))
            clone = stack.enter_context(game.clone())
            native = _engine.NativeOpponents(game._native,0,json.dumps([None,{"type":"random","seed":1}]))
            before = game.position
            with self.assertRaises(ValueError): native.apply_candidate(game._native,79,before)
            self.assertEqual(game.position,before)
            with self.assertRaises(tag.StalePositionError): native.advance(clone._native)
            native.apply_candidate(game._native,0,before)
            with self.assertRaises(tag.StalePositionError): native.apply_candidate(game._native,0,before)
            game.reset()
            with self.assertRaises(tag.StalePositionError): native.advance(game._native)


@unittest.skipUnless(HAS_TORCH,"optional PyTorch not installed")
class MultiplayerEvaluation(unittest.TestCase):
    def settings(self,players=3,**changes):
        evaluation = {"enabled":True,"games":players,"opponent":{"name":"random","agent":{"type":"random"}},
                      "ranking":"score","include_actions":True,"max_actions":1000}
        evaluation.update(changes)
        return stochastic_settings(players,evaluation=evaluation)

    def test_every_player_count_rotates_and_replays_reports_without_modifying_training(self):
        for players in (2,3,4):
            settings = self.settings(players)
            session = SelfPlaySession(settings)
            weights = {k:v.clone() for k,v in session.model.state_dict().items()}
            rng = copy.deepcopy(session.sample_rng.bit_generator.state)
            torch_rng = session.torch_rng.clone()
            report = evaluate_model(settings,FrozenModel(session.model))
            self.assertFalse(report["failures"])
            self.assertEqual([r["games"] for r in report["per_seat"]],[1]*players)
            self.assertEqual(report["completed_games"],players)
            places = []
            for row in report["games"]:
                self.assertEqual(row["lineup"][row["candidate_seat"]]["name"],"candidate")
                with tag.create("cantstop",dict(settings.parameters,seed=row["game_seed"]),players=players) as game:
                    for event in row["actions"]:
                        self.assertEqual(game.next_player,event["actor"])
                        game.apply(event["action_id"],position=game.position)
                        if event["dice"] is not None:
                            actual=game.observe(0,("dice",)).payloads["dice"].argmax(axis=1).tolist()
                            self.assertEqual(actual,event["dice"])
                    self.assertIsNone(game.next_player)
                    self.assertEqual(game.observe(0,()).results.tolist(),row["outcomes"])
                    scores=game.observe(0,("owners",)).payloads["owners"][:,1:].sum(axis=0).tolist()
                    self.assertEqual(scores,row["scores"])
                expected = [sum(s>score for s in scores)+1+(scores.count(score)-1)/2 for score in scores]
                self.assertEqual(row["finishing_positions"],expected)
                places.append(expected[row["candidate_seat"]])
                self.assertEqual(row["moves"],len(row["actions"]))
            self.assertEqual(report["combined"]["average_finishing_position"],sum(places)/players)
            wins=sum(row["outcomes"][row["candidate_seat"]]==1 for row in report["games"])
            self.assertEqual(report["combined"]["wins"],wins)
            self.assertEqual(report["combined"]["win_rate"],wins/players)
            self.assertTrue(all(torch.equal(v,session.model.state_dict()[k]) for k,v in weights.items()))
            self.assertEqual(rng,session.sample_rng.bit_generator.state)
            self.assertTrue(torch.equal(torch_rng,session.torch_rng))
            self.assertEqual(session.completed_cycles,0)
            self.assertEqual(len(session.buffer),0)
            before=settings.document()
            report['candidate_search']['seed']=999
            report['opponents'][0]['agent']['type']='changed'
            self.assertEqual(settings.document(),before)

    def test_mixed_lineups_fixed_seats_and_rotations_continue_across_cycles(self):
        entries=[{"name":"random","agent":{"type":"random"}},
                 {"name":"osla","agent":{"type":"osla","repetitions_per_action":2}},
                 {"name":"mcts","agent":{"type":"mcts","config":{"schema_version":1,"budgets":{"iterations":3},"rollout_length":0}}}]
        settings=self.settings(4,opponent=None,opponents=entries,games=5)
        session=SelfPlaySession(settings)
        snapshot=FrozenModel(session.model)
        report=evaluate_model(settings,snapshot,cycle=1)
        self.assertEqual([r["candidate_seat"] for r in report["games"]],[1,2,3,0,1])
        for row in report["games"]:
            self.assertEqual([e["name"] for e in row["lineup"] if e["name"]!="candidate"],["random","osla","mcts"])
        fixed=replace(settings,evaluation=replace(settings.evaluation,positions="fixed",candidate_seat=3,games=2))
        fixed_report=evaluate_model(fixed,snapshot)
        self.assertEqual([r["candidate_seat"] for r in fixed_report["games"]],[3,3])
        self.assertEqual([r["games"] for r in fixed_report["per_seat"]],[0,0,0,2])

    def test_truncations_are_not_outcomes_and_terminal_action_at_limit_completes(self):
        settings=self.settings(max_actions=1)
        session=SelfPlaySession(settings)
        snapshot=FrozenModel(session.model)
        limited=evaluate_model(settings,snapshot)
        self.assertEqual(limited["truncated_games"],3)
        self.assertEqual(limited["completed_games"],0)
        self.assertEqual(limited["combined"]["wins"],0)
        self.assertIsNone(limited["combined"]["win_rate"])
        self.assertIsNone(limited["combined"]["average_finishing_position"])
        self.assertTrue(all(row["outcomes"] is None and "finishing_positions" not in row for row in limited["games"]))
        full_settings=replace(settings,evaluation=replace(settings.evaluation,games=1,max_actions=1000))
        full=evaluate_model(full_settings,snapshot)
        exact_settings=replace(full_settings,evaluation=replace(full_settings.evaluation,max_actions=full["games"][0]["moves"]))
        exact=evaluate_model(exact_settings,snapshot)
        self.assertEqual(exact["games"][0]["outcomes"],full["games"][0]["outcomes"])
        self.assertEqual(exact["completed_games"],1)

    def test_automatic_seeds_repeat_and_runtime_failure_keeps_completed_peers(self):
        settings=self.settings()
        session=SelfPlaySession(settings)
        snapshot=FrozenModel(session.model)
        with patch("tag_rust._engine.automatic_search_seed",return_value=123) as source:
            auto=evaluate_model(replace(settings,seed=None),snapshot)
            source.assert_called_once()
        explicit=evaluate_model(replace(settings,seed=123),snapshot)
        self.assertEqual(auto["games"],explicit["games"])
        from tag_rust.learning.selfplay import lineup_evaluation
        original=lineup_evaluation.start_search
        def fail_second(game,settings,configuration,evaluator):
            if game.seed==explicit["games"][1]["game_seed"]:
                raise RuntimeError("inference failed")
            return original(game,settings,configuration,evaluator)
        with patch.object(lineup_evaluation,"start_search",side_effect=fail_second):
            failed=evaluate_model(replace(settings,seed=123),snapshot)
        self.assertEqual(failed["completed_games"],1)
        self.assertEqual(failed["failed_games"],1)
        self.assertEqual(failed["unplayed_games"],1)
        self.assertEqual(failed["games"][0],explicit["games"][0])

    def test_tied_finishing_places_and_configuration_errors(self):
        self.assertEqual(finishing_positions([3,1,1,0]),[1,2.5,2.5,4])
        for change in ({"opponents":[]}, {"opponents":[{"name":"x","agent":{"type":"random"}}],"opponent":None},
                       {"max_actions":0}, {"ranking":"winner"},
                       {"opponent":{"name":"candidate","agent":{"type":"random"}}},
                       {"opponent":{"name":"x","agent":{"type":"osla","repetitions_per_action":0}}}):
            with self.assertRaises(ValueError): self.settings(**change)
        invalid=self.settings(opponent={"name":"x","agent":{"type":"mcts","config":{
            "schema_version":1,"state_evaluator":{"name":"missing"}}}})
        with self.assertRaises(ValueError): SelfPlaySession(invalid)

    def test_post_cycle_evaluation_and_checkpoint_resume_keep_training_identical(self):
        settings=self.settings(games=3,max_actions=1)
        session=SelfPlaySession(settings)
        first=session.run_cycle()
        self.assertEqual(first["evaluation"]["truncated_games"],3)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"state.pt"
            save_checkpoint(path,session)
            resumed=load_checkpoint(path)
            left,right=session.run_cycle(),resumed.run_cycle()
            self.assertEqual(left["model_identity"],right["model_identity"])
            self.assertEqual(left["evaluation"]["games"],right["evaluation"]["games"])
            self.assertEqual(left["optimization"],right["optimization"])

    def test_standalone_model_cli_infers_contract_and_records_owned_report(self):
        from tag_rust.learning.artifacts import save_model
        settings=self.settings(games=3,max_actions=1)
        session=SelfPlaySession(settings)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            model=root/"model.json"
            config=root/"evaluation.json"
            report=root/"report.json"
            save_model(model,session.model)
            # Standalone evaluation reads tensor/value descriptions from the model.
            document={"schema_version":1,"game":"cantstop","players":3,"seed":42,
                "parameters":settings.parameters,"search":settings.search,
                "evaluation":{"games":3,"opponent":{"name":"r","agent":{"type":"random"}},"max_actions":1},
                "output":{"report_file":"report.json"}}
            config.write_text(json.dumps(document))
            command=[sys.executable,str(ADAPTER/'examples/selfplay_evaluate.py'),
                     '--config',str(config),'--model',str(model)]
            completed=subprocess.run(command,text=True,capture_output=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            saved=json.loads(report.read_text())
            self.assertEqual(saved['evaluation']['truncated_games'],3)
            self.assertEqual(saved['evaluation']['model_identity'],FrozenModel(session.model).identity)
            self.assertEqual(json.loads(completed.stdout.splitlines()[0])['master_seed'],42)
            original=report.read_bytes()
            document['players']=4
            config.write_text(json.dumps(document))
            failed=subprocess.run(command,text=True,capture_output=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(report.read_bytes(),original)


if __name__=="__main__":
    unittest.main()
