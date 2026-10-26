"""Independent-game batching preserves decisions and validates whole responses."""
from collections import deque
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
import tag_rust as tag
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from open_lines import evaluate_open_lines
import test_execution as execution
setup, CASES = execution.setup, execution.CASES


class GroupingTests(unittest.TestCase):
    assert_same_game = execution.ExecutionTests.assert_same_game

    def test_grouped_matches_each_move_and_rng(self):
        for rows in (1, 3, 1024):
            with self.subTest(rows=rows), ExitStack() as stack:
                games = [stack.enter_context(setup(n,p,native=False,rows=rows)) for n,p in CASES]
                # Add another instance of each game so compatibility merging is exercised.
                games += [stack.enter_context(setup(n,p,native=False,rows=rows)) for n,p in CASES]
                refs = [stack.enter_context(g.clone()) for g in games]
                scorers = [g.open_lines_evaluator() for g in refs]
                lines = {}
                for g in games:
                    spec = g.spec()
                    lines[spec.identity] = spec.heuristic.indices.reshape(-1,spec.heuristic.window_length)
                batch = stack.enter_context(tag.GameBatch(games))
                rounds = 0
                while any(g.next_player is not None for g in games):
                    queue = deque(batch.advance(max_moves_per_game=1).groups)
                    while queue:
                        group = queue.popleft()
                        self.assertEqual(group.payloads["cells"].shape[0],len(group.action_ids))
                        for segment in group.segments:
                            sl = slice(segment.start,segment.stop)
                            self.assertTrue(np.all(group.game_indices[sl] == segment.game_index))
                            self.assertTrue(np.all(group.observers[sl] == segment.actor))
                        outcome = batch.resume(group,evaluate_open_lines(group,lines[group.identity]),max_moves_per_game=1)
                        for stop in outcome.stops:
                            self.assertIsNone(stop.error)
                            if stop.moves:
                                i = stop.game_index
                                result = refs[i].advance(max_moves=1)
                                while result.reason == "evaluation":
                                    result = refs[i].resume(result.request,scorers[i].evaluate(result.request),max_moves=1)
                                self.assertEqual(result.moves,1)
                            self.assert_same_game(games[stop.game_index],refs[stop.game_index])
                        queue.extend(outcome.groups)
                    rounds += 1
                    self.assertLess(rounds,100)
                self.assertEqual(batch.pending_count,0)

    def test_mixed_players_settings_and_payloads(self):
        with ExitStack() as stack:
            games = [stack.enter_context(tag.create("tictactoe")) for _ in range(5)]
            games[1].apply(0,position=games[1].position)
            for i,g in enumerate(games):
                for seat in range(2):
                    g.set_controller(seat,tag.OSLAPython(evaluator_id="test",seed=seat,
                        evaluator_parameters={"mode":1 if i==3 else 0},
                        payloads=("board",) if i==4 else ("cells",)))
            references = [stack.enter_context(g.clone()) for g in games]
            batch = stack.enter_context(tag.GameBatch(games,max_group_rows=27))
            result = batch.advance()
            self.assertEqual(len(result.groups),3)
            group = result.groups[0]
            self.assertEqual([s.game_index for s in group.segments],[0,1,2])
            self.assertEqual([s.actor for s in group.segments],[0,1,0])
            support = games[0].spec().heuristic
            values = evaluate_open_lines(group,support.indices.reshape(-1,support.window_length))
            for segment in group.segments:
                ref = references[segment.game_index]
                individual = ref.prepare_evaluation()
                sl = slice(segment.start,segment.stop)
                np.testing.assert_array_equal(group.payloads["cells"][sl],individual.payloads["cells"])
                np.testing.assert_array_equal(values[sl],ref.open_lines_evaluator().evaluate(individual))
                # Compare combined rows with a fresh manual request on an identical game.
                source = games[segment.game_index]
                view = source.observe(segment.actor,payloads=("cells",))
                np.testing.assert_array_equal(group.action_ids[segment.start:segment.stop],view.offered_action_ids)
            batch.resume(group,np.zeros(len(group.action_ids)),max_moves_per_game=1)
            self.assertEqual([int(g.observe(0).progress[0]) for g in games],[1,2,1,0,0])

    def test_invalid_group_is_retryable_without_any_moves_or_rng(self):
        with ExitStack() as stack:
            games = [stack.enter_context(setup(native=False)) for _ in range(3)]
            batch = stack.enter_context(tag.GameBatch(games))
            group, = batch.advance().groups
            before = [(g.position,g.controller_checkpoint(0)) for g in games]
            values = np.zeros(len(group.action_ids))
            bad = values.copy(); bad[-1] = np.nan
            ids = group.action_ids.copy(); ids[-1] += 1
            attempts = [(bad,{}),(values[:-1],{}),(values,{"action_ids":ids}),
                        (values,{"evaluator":replace(group.evaluator,configuration='{"changed":true}')})]
            for value,kwargs in attempts:
                with self.assertRaises(ValueError): batch.resume(group,value,**kwargs)
                self.assertEqual([(g.position,g.controller_checkpoint(0)) for g in games],before)
                self.assertEqual(batch.pending_count,1)
            batch.resume(group,values,max_moves_per_game=1)
            self.assertTrue(all(g.observe(0).progress[0] == 1 for g in games))
            with self.assertRaises(tag.StalePositionError): batch.resume(group,values)

    def test_cancel_after_partial_chunks_restarts_without_rng_changes(self):
        with ExitStack() as stack:
            games = [stack.enter_context(setup(native=False,rows=3)) for _ in range(2)]
            batch = stack.enter_context(tag.GameBatch(games))
            before = [(g.position,g.controller_checkpoint(0)) for g in games]
            first, = batch.advance().groups
            second, = batch.resume(first,np.zeros(len(first.action_ids))).groups
            self.assertEqual([s.offset for s in second.segments],[3,3])
            invalid = np.zeros(len(second.action_ids)); invalid[-1] = np.inf
            with self.assertRaises(ValueError): batch.resume(second,invalid)
            self.assertEqual([(g.position,g.controller_checkpoint(0)) for g in games],before)
            batch.cancel_all()
            fresh, = batch.advance().groups
            self.assertEqual([s.offset for s in fresh.segments],[0,0])
            np.testing.assert_array_equal(fresh.payloads["cells"],first.payloads["cells"])
            np.testing.assert_array_equal(fresh.action_ids,first.action_ids)
            self.assertEqual([(g.position,g.controller_checkpoint(0)) for g in games],before)
            with self.assertRaises(tag.StalePositionError): batch.resume(second,np.zeros(len(second.action_ids)))

    def test_stale_member_rejects_all_and_cancel_preserves_new_work(self):
        with ExitStack() as stack:
            games = [stack.enter_context(setup(native=False)) for _ in range(2)]
            batch = stack.enter_context(tag.GameBatch(games))
            group, = batch.advance().groups
            before = games[0].controller_checkpoint(0)
            games[1].reset()
            new_request = games[1].prepare_evaluation()
            with self.assertRaises(tag.StalePositionError): batch.resume(group,np.zeros(len(group.action_ids)))
            self.assertEqual(games[0].controller_checkpoint(0),before)
            batch.cancel(group)
            # Cancelling the obsolete group leaves the replacement request intact.
            games[1].finish_evaluation(new_request,np.zeros(len(new_request.action_ids)))
            games[1].cancel_evaluation(new_request)
            self.assertEqual(batch.pending_count,0)
            again = batch.advance().groups[0]
            batch.cancel_all()
            with self.assertRaises(tag.StalePositionError): batch.resume(again,np.zeros(len(again.action_ids)))
            self.assertEqual(games[0].controller_checkpoint(0),before)

    def test_limits_duplicates_foreign_handles_and_original_lifetime(self):
        with ExitStack() as stack:
            a,b = [stack.enter_context(setup(native=False)) for _ in range(2)]
            with self.assertRaises(ValueError): tag.GameBatch([a,a])
            for kwargs in ({"max_group_rows":8},{"max_bytes":100}):
                with tag.GameBatch([a,b],**kwargs) as batch:
                    before = (a.position,b.position)
                    with self.assertRaises(MemoryError): batch.advance()
                    self.assertEqual((a.position,b.position),before)
            with tag.GameBatch([a],max_group_rows=9) as one, tag.GameBatch([b]) as two:
                group, = one.advance().groups
                with self.assertRaises(tag.StalePositionError): two.resume(group,np.zeros(len(group.action_ids)))
                with self.assertRaises(tag.EvaluationError): one.advance()
            # Closing a coordinator cancels its work but leaves original games usable.
            self.assertEqual(a.advance().reason,"evaluation")

    def test_native_human_errors_and_closed_members(self):
        with ExitStack() as stack:
            native = stack.enter_context(setup())
            human = stack.enter_context(tag.create("connect4"))
            broken = stack.enter_context(setup())
            before = broken.position
            broken.set_controller(1,tag.OSLAPython(evaluator_id="bad",seed=1,max_candidates=1))
            with tag.GameBatch([native,human,broken]) as batch:
                result = batch.advance()
                self.assertEqual([s.reason for s in result.stops],["finished","human","error"])
                self.assertGreater(result.stops[0].moves,0)
                self.assertEqual(native.next_player,None)
                self.assertEqual(result.stops[2].moves,1)
                self.assertEqual(broken.position.revision,before.revision+2)
                self.assertEqual(broken.observe(0).progress[0],1)
            a,b = [stack.enter_context(setup(native=False)) for _ in range(2)]
            with tag.GameBatch([a,b]) as batch:
                group, = batch.advance().groups
                b.close()
                with self.assertRaises(tag.ClosedGameError): batch.resume(group,np.zeros(len(group.action_ids)))
                self.assertEqual(a.observe(0).progress[0],0)
                batch.cancel(group)

    def test_row_cap_separates_groups_and_returned_arrays_own_storage(self):
        with ExitStack() as stack:
            games = [stack.enter_context(setup(native=False)) for _ in range(3)]
            batch = stack.enter_context(tag.GameBatch(games,max_group_rows=18))
            result = batch.advance()
            self.assertEqual([len(g.action_ids) for g in result.groups],[18,9])
            saved = result.groups[0].payloads["cells"].copy()
            batch.close()
            for g in games: g.close()
            np.testing.assert_array_equal(result.groups[0].payloads["cells"],saved)


if __name__ == "__main__":
    unittest.main()
