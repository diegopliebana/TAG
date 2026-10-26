"""Rust owns execution; Python supplies values only at explicit evaluation stops."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import unittest
import numpy as np
import tag_rust as tag

ADAPTER = Path(__file__).resolve().parents[1]
CASES = [("tictactoe",{}),("tictactoe",{"grid_size":5}),
         ("tictactoe",{"grid_size":1}),("connect4",{}),
         ("connect4",{"width":4,"height":3,"winning_length":3}),
         ("connect4",{"width":3,"height":2,"winning_length":1})]


def setup(name="tictactoe", parameters=None, *, native=True, rows=1024):
    game = tag.create(name,parameters)
    for seat in range(2):
        controller = (tag.OSLAPython.rust_heuristic(seed=7+seat,max_rows=rows) if native
                      else tag.OSLAPython(evaluator_id="rust-open-lines",seed=7+seat,max_rows=rows))
        game.set_controller(seat,controller)
    return game


def manual_move(game, scorer):
    request = game.prepare_evaluation()
    while isinstance(request,tag.EvaluationRequest):
        request = game.finish_evaluation(request,scorer.evaluate(request))
    game.commit_evaluation(request)


class ExecutionTests(unittest.TestCase):
    def assert_same_game(self,a,b):
        for seat in range(2):
            x,y = a.observe(seat),b.observe(seat)
            for key in ("progress","results","offered_action_ids","action_mask"):
                np.testing.assert_array_equal(getattr(x,key),getattr(y,key))
            for name in ("cells","board"):
                np.testing.assert_array_equal(x.payloads[name],y.payloads[name])
            self.assertEqual(x.next_player,y.next_player)
            self.assertEqual(replace(a.controller_checkpoint(seat),execution="external"),
                             replace(b.controller_checkpoint(seat),execution="external"))

    def test_native_advance_matches_every_manual_move_and_rng(self):
        for name,parameters in CASES:
            for rows in (1,3,1024):
                with self.subTest(game=name,parameters=parameters,rows=rows):
                    with setup(name,parameters,rows=rows) as owned, setup(name,parameters,native=False,rows=rows) as manual:
                        scorer = manual.open_lines_evaluator()
                        while manual.next_player is not None:
                            previous = owned.position
                            result = owned.advance(max_moves=1)
                            manual_move(manual,scorer)
                            self.assertEqual(result.moves,1)
                            self.assertEqual(result.position,owned.position)
                            self.assertNotEqual(previous,owned.position)
                            self.assertEqual(result.reason,"finished" if manual.next_player is None else "move_limit")
                            self.assertIsNone(result.request)
                            self.assert_same_game(owned,manual)
                        ended = owned.advance()
                        self.assertEqual((ended.reason,ended.moves,ended.actor),("finished",0,None))

    def test_external_resume_chunks_limits_and_automatic_commit(self):
        for name,parameters in CASES:
            with setup(name,parameters,native=False,rows=3) as game, setup(name,parameters,rows=3) as reference:
                expected = reference.advance(max_moves=4096)
                self.assertEqual(expected.reason,"finished")
                scorer = game.open_lines_evaluator()
                result = game.advance(max_moves=1)
                accepted = 0
                while result.reason != "finished":
                    if result.reason == "move_limit":
                        result = game.advance(max_moves=1)
                        self.assertEqual(result.moves,0)
                        continue
                    self.assertEqual(result.reason,"evaluation")
                    request = result.request
                    before = game.controller_checkpoint(request.actor)
                    old_position = game.position
                    result = game.resume(request,scorer.evaluate(request),max_moves=1)
                    accepted += result.moves
                    if result.moves == 0:
                        self.assertEqual(game.position,old_position)
                        self.assertEqual(game.controller_checkpoint(request.actor),before)
                    else:
                        self.assertEqual(result.moves,1)
                        self.assertNotEqual(game.position,old_position)
                self.assertEqual(accepted,expected.moves)
                self.assert_same_game(game,reference)

    def test_mixed_seats_and_human_stop_precede_move_limit(self):
        with tag.create("tictactoe") as game:
            self.assertEqual((game.advance().reason,game.advance().actor),("human",0))
            game.set_controller(0,tag.OSLAPython.rust_heuristic(seed=7))
            result = game.advance(max_moves=1)
            self.assertEqual((result.reason,result.actor,result.moves),("human",1,1))
            game.apply(0,position=result.position)
            self.assertEqual(game.advance().moves,1)
        with setup(native=False) as game:
            game.set_controller(1,tag.OSLAPython.rust_heuristic(seed=8))
            result = game.advance()
            scorer = game.open_lines_evaluator()
            result = game.resume(result.request,scorer.evaluate(result.request),max_moves=2)
            self.assertEqual((result.reason,result.actor,result.moves),("move_limit",0,2))
            self.assertEqual(game.observe(0).progress[0],2)

    def test_retries_cancellation_stale_handles_and_budget_validation(self):
        with setup(native=False) as game:
            request = game.advance().request
            scorer = game.open_lines_evaluator()
            checkpoint = game.controller_checkpoint(0)
            with self.assertRaises(tag.EvaluationError):
                game.advance()
            with self.assertRaises(tag.EvaluationError):
                game.clone()
            for budget in (0,-1,True):
                with self.assertRaises((ValueError,TypeError)):
                    game.resume(request,scorer.evaluate(request),max_moves=budget)
            with self.assertRaises(ValueError):
                game.resume(request,np.full(request.action_ids.size,np.nan,dtype=np.float64))
            with self.assertRaises(TypeError):
                game.resume(request,np.zeros(request.action_ids.size,dtype=np.float32))
            with self.assertRaises(ValueError):
                game.resume(request,scorer.evaluate(request),action_ids=request.action_ids[::-1].copy())
            self.assertEqual(game.controller_checkpoint(0),checkpoint)
            game.cancel_evaluation(request)
            replacement = game.advance().request
            np.testing.assert_array_equal(request.payloads["cells"],replacement.payloads["cells"])
            with self.assertRaises(tag.StalePositionError):
                game.resume(request,scorer.evaluate(request))
            result = game.resume(replacement,scorer.evaluate(replacement),max_moves=1)
            self.assertEqual((result.reason,result.moves),("move_limit",1))
            with self.assertRaises(tag.StalePositionError):
                game.resume(replacement,scorer.evaluate(replacement))
            request = game.advance().request
            game.reset()
            with self.assertRaises(tag.StalePositionError):
                game.resume(request,scorer.evaluate(request))
            for budget in (0,-1,True):
                with self.assertRaises((ValueError,TypeError)):
                    game.advance(max_moves=budget)
        with self.assertRaises(tag.ClosedGameError):
            game.advance()

    def test_native_checkpoints_clone_reset_and_invalid_backend(self):
        with setup() as game:
            with self.assertRaisesRegex(ValueError,"advance"):
                game.prepare_evaluation()
            self.assertEqual(game.advance(max_moves=3).moves,3)
            with game.clone() as copy:
                for seat in range(2):
                    saved = game.controller_checkpoint(seat)
                    self.assertEqual(saved.execution,"rust_open_lines")
                    copy.restore_controller(seat,saved)
                self.assertEqual(game.advance().reason,"finished")
                self.assertEqual(copy.advance().reason,"finished")
                self.assert_same_game(game,copy)
                game.reset()
                copy.reset()
                game.advance()
                copy.advance()
                self.assert_same_game(game,copy)
            saved = game.controller_checkpoint(0)
            for bad in (replace(saved,execution="unknown"),replace(saved,evaluator_version=99),
                        replace(saved,evaluator_configuration='{"changed":true}')):
                before = game.position
                with self.assertRaises(ValueError):
                    game.restore_controller(0,bad)
                self.assertEqual(game.position,before)
            with self.assertRaises(ValueError):
                game.prepare_evaluation()
            legacy = asdict(saved)
            legacy.pop("execution")
            self.assertEqual(tag.ControllerCheckpoint(**legacy).execution,"external")

    def test_error_after_native_move_keeps_position_and_recovery(self):
        with setup() as game:
            game.set_controller(1,tag.OSLAPython(evaluator_id="limit-test",seed=8,max_candidates=1))
            before = game.position
            with self.assertRaises(MemoryError):
                game.advance(max_moves=10)
            self.assertEqual(game.position.revision,before.revision+1)
            self.assertEqual(game.observe(0).progress[0],1)
            self.assertEqual(game.next_player,1)
            with game.clone() as copy:
                self.assertEqual(copy.observe(0).progress[0],1)
            game.set_controller(1,tag.OSLAPython.rust_heuristic(seed=8))
            self.assertEqual(game.advance().reason,"finished")

    def test_requests_after_native_progress_foreign_handles_and_replacement(self):
        with setup() as game:
            game.set_controller(1,tag.OSLAPython(evaluator_id="rust-open-lines",seed=8))
            result = game.advance()
            self.assertEqual((result.reason,result.actor,result.moves),("evaluation",1,1))
            self.assertEqual(result.request.position,game.position)
            scorer = game.open_lines_evaluator()
            result = game.resume(result.request,scorer.evaluate(result.request),max_moves=3)
            self.assertEqual((result.reason,result.actor,result.moves),("evaluation",1,2))
            self.assertEqual(result.request.position,game.position)
            game.cancel_evaluation(result.request)
            with game.clone() as other:
                request = game.advance().request
                foreign = other.advance().request
                with self.assertRaises(tag.StalePositionError):
                    other.resume(request,scorer.evaluate(request))
                self.assertEqual(other.position,foreign.position)
                game.set_controller(1,None)
                with self.assertRaises(tag.StalePositionError):
                    game.resume(request,scorer.evaluate(request))
                self.assertEqual(game.advance().reason,"human")

    def test_native_reference_owned_runs_match_python_calls(self):
        output = subprocess.run(["cargo","run","--offline","--quiet","--manifest-path",
            str(ADAPTER/"Cargo.toml"),"--example","native_reference"],check=True,capture_output=True,text=True)
        for record in json.loads(output.stdout):
            case = record["case"]
            for expected in record["owned_runs"]:
                with setup(case["game"],case["parameters"],native=expected["native"],rows=3) as game:
                    result = game.advance(max_moves=2)
                    moves = 0
                    while True:
                        moves += result.moves
                        if result.reason == "finished": break
                        if result.reason == "move_limit": result = game.advance(max_moves=2)
                        else:
                            request = result.request
                            result = game.resume(request,np.zeros(request.action_ids.size,dtype=np.float64),max_moves=2)
                    self.assertEqual(moves,expected["moves"])
                    actual = game.observe(0,payloads=("cells",))
                    np.testing.assert_array_equal(actual.payloads["cells"],expected["final"]["payloads"]["cells"]["values"])
                    np.testing.assert_array_equal(actual.results,expected["final"]["results"])
                    self.assertEqual(game.controller_checkpoint(0).execution,expected["mode"])
