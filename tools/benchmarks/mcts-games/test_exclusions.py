"""Verify interruption exclusions against complete small, recorded control shards."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import summarize
import run

CONTROL=Path(__file__).resolve().parents[3]/'.scratch/mcts-games-validation-final/stride-1024'

class ExclusionTests(unittest.TestCase):
    def test_baseline_saved_before_games_and_resume_keeps_game_and_process_baselines(self):
        scratch=Path(__file__).resolve().parents[3]/'.scratch'
        with tempfile.TemporaryDirectory(prefix='mcts-resume-test-',dir=scratch) as temporary:
            folder=Path(temporary);destination=folder/'batch.json';probe=folder/'fake.py'
            probe.write_text('''import json,sys,pathlib
p=pathlib.Path(__file__).parent
print(json.dumps({'phase':'ready','timer_control_ns':1}),flush=True)
input()
assert (p/'batch.baseline.json').exists(), 'baseline must be saved before first game'
for i in range(int(sys.argv[3])):
 print(json.dumps({'phase':'game','id':i,'seed':int(sys.argv[5])+i,'elapsed_ns':1,'decisions':1}),flush=True)
 input()
 if not (p/'failed').exists():
  (p/'failed').touch()
  sys.exit(7)
''')
            identity={'engine':'rust','game':'tictactoe','players':2,'shard':0,'games':2,'seed':42,'settings':{}}
            command=[sys.executable,str(probe),'tictactoe','2','2','5000','42','1024','true','true','6']
            memory={'working_set':100,'private_bytes':200}
            with patch.object(run.research,'process_memory',return_value=memory),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'Probe exit 7'):
                    run.run_process(command,destination,identity)
                original=destination.with_suffix('.progress.jsonl').read_bytes()
                # Incomplete final writes may be trimmed; complete records stay.
                with destination.with_suffix('.progress.jsonl').open('ab') as stream:stream.write(b'{"id":')
                result=run.run_process(command,destination,identity)
            self.assertEqual([g['id'] for g in result['games']],[0,1])
            self.assertEqual([g['seed'] for g in result['games']],[42,43])
            self.assertEqual([g['process_segment'] for g in result['games']],[0,1])
            self.assertEqual(len(result['process_segments']),2)
            self.assertTrue(destination.with_suffix('.progress.jsonl').read_bytes().startswith(original))
            with patch.object(run.subprocess,'Popen') as launch,contextlib.redirect_stdout(io.StringIO()):
                run.run_process(command,destination,identity)
                launch.assert_not_called()

    def test_whole_game_exclusion_filters_all_statistics_and_keeps_raw_records(self):
        self.assertTrue(CONTROL.exists(),'Run the probe selftest before this integration check')
        scratch=Path(__file__).resolve().parents[3]/'.scratch'
        with tempfile.TemporaryDirectory(prefix='mcts-exclusion-test-',dir=scratch) as temporary:
            root=Path(temporary)
            (root/'settings.json').write_bytes((CONTROL/'settings.json').read_bytes())
            for path in CONTROL.glob('*-shard*.json'):
                (root/path.name).write_bytes(path.read_bytes())
            target=root/'java-loveletter-3-shard0.json'
            raw=json.loads(target.read_text());expected=copy.deepcopy(raw['games'][0])
            bad=raw['games'][1]
            bad['elapsed_ns']=10**18;bad['search_ns']=10**18;bad['probe_ns']=10**18
            for metric in bad['metrics']:metric['ns']=10**18
            for field in ('working_set','private_bytes'):bad['memory_peak'][field]=2**50
            target.write_text(json.dumps(raw));before=target.read_bytes()
            # An equally slow unlisted game must remain: no automatic outlier rule.
            other=root/'rust-tictactoe-2-shard0.json';unlisted=json.loads(other.read_text())
            unlisted['games'][1]['elapsed_ns']=10**18
            unlisted['process_segments']=[{'baseline':unlisted['baseline']},{'baseline':{'working_set':123,'private_bytes':456}}]
            for index,game in enumerate(unlisted['games']):game['process_segment']=index
            other.write_text(json.dumps(unlisted))
            other.with_suffix('.baseline.json').write_text(json.dumps({'segments':unlisted['process_segments']}))
            (root/'exclusions.json').write_text(json.dumps({'version':1,'games':[
                {'engine':'java','game':'loveletter','players':3,'shard':0,'id':1,'seed':bad['seed'],'reason':'Synthetic sleep interruption for test only'}]}))
            with contextlib.redirect_stdout(io.StringIO()):summarize.summarize(root)
            result=json.loads((root/'validation.json').read_text())
            self.assertEqual((result['games'],result['included_games'],result['excluded_games']),(32,31,1))
            work=json.loads((root/'game-summary.json').read_text())
            kept=next(r for r in work if (r['engine'],r['game'],r['players'])==('java','loveletter',3))
            self.assertEqual((kept['collected_games'],kept['games'],kept['excluded_games']),(2,1,1))
            self.assertEqual(kept['mean_game_seconds'],expected['elapsed_ns']/1e9)
            self.assertEqual(kept['peak_working_set'],expected['memory_peak']['working_set'])
            api=json.loads((root/'api-summary.json').read_text())
            for row in (r for r in api if (r['engine'],r['game'],r['players'])==('java','loveletter',3)):
                original=next(m for m in expected['metrics'] if (m['scope'],m['operation'])==(row['scope'],row['operation']))
                self.assertEqual(row['sampled_ns'],original['ns'])
                self.assertEqual(row['calls'],original['calls'])
            self.assertGreater(next(r for r in work if (r['engine'],r['game'])==('rust','tictactoe'))['mean_game_seconds'],1e8)
            self.assertEqual(target.read_bytes(),before)
            memory=json.loads((root/'process-memory-summary.json').read_text())
            segments=[r for r in memory if (r['engine'],r['game'])==('rust','tictactoe')]
            self.assertEqual(len(segments),2)
            self.assertEqual(segments[1]['baseline_private_bytes'],456)

    def test_wrong_seed_cannot_exclude_a_different_game(self):
        result={'identity':{'engine':'java','game':'loveletter','players':3,'shard':0},'games':[{'id':39,'seed':81}]}
        key=('java','loveletter',3,0,39)
        with self.assertRaisesRegex(AssertionError,'seed'):
            summarize.included_games(result,{key:{'seed':78}})

    def test_resizing_reuses_completed_batches_but_rejects_changed_search_settings(self):
        scratch=Path(__file__).resolve().parents[3]/'.scratch'
        old={'engine':'rust','game':'tictactoe','players':2,'shard':0,'games':100,'seed':42,
             'settings':{'games_per_profile':500,'shards':5,'iterations':5000,'seed':42}}
        new=copy.deepcopy(old);new['settings'].update(games_per_profile=250,shards=3,shard_sizes=[100,100,50])
        self.assertEqual(run.shard_sizes(new['settings']),[100,100,50])
        with tempfile.TemporaryDirectory(prefix='mcts-resize-test-',dir=scratch) as temporary:
            path=Path(temporary)/'completed.json'
            saved={'identity':old,'games':[{'id':i} for i in range(100)]}
            path.write_text(json.dumps(saved));before=path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.run_process(['must-not-start'],path,new),saved)
            self.assertEqual(path.read_bytes(),before)
            wrong=copy.deepcopy(new);wrong['settings']['iterations']=1000
            with self.assertRaisesRegex(ValueError,'settings differ'):
                run.run_process(['must-not-start'],path,wrong)

    def test_batch_boundary_gate_prevents_launch_without_creating_a_fake_result(self):
        scratch=Path(__file__).resolve().parents[3]/'.scratch'
        with tempfile.TemporaryDirectory(prefix='mcts-gate-test-',dir=scratch) as temporary:
            destination=Path(temporary)/'next.json'
            gate=destination.with_suffix('.stderr.log');gate.mkdir()
            with patch.object(run.subprocess,'Popen') as launch:
                with self.assertRaises(OSError):run.run_process(['must-not-start'],destination,{})
                launch.assert_not_called()
            self.assertFalse(destination.exists())

    def test_pooled_means_use_250_or_actual_included_games_not_equal_batch_weights(self):
        rows=[{'games':100,'total_elapsed_ns':200*10**9},
              {'games':100,'total_elapsed_ns':400*10**9},
              {'games':50,'total_elapsed_ns':500*10**9}]
        self.assertEqual(summarize.pooled_game_mean(rows,'mean_game_seconds'),1100/250)
        rows[0]={'games':98,'total_elapsed_ns':196*10**9}
        self.assertEqual(summarize.pooled_game_mean(rows,'mean_game_seconds'),1096/248)
        self.assertEqual(summarize.pooled_api_mean([{'samples':100,'sampled_ns':1000},
            {'samples':10,'sampled_ns':1000}]),2000/110)

if __name__=='__main__':unittest.main()
