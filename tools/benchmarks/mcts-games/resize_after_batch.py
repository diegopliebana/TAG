"""Finish the active batch, gate the old collector before its next launch, and resize."""
import ctypes
from ctypes import wintypes
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import run

ROOT=run.ROOT
OUTPUT=ROOT/'docs/benchmarks/research/mcts-games'
STATUS=ROOT/'.scratch/mcts-games-250-transition.json'

def stamp(state,**extra):
    STATUS.write_text(json.dumps({'state':state,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra},indent=2)+'\n')

def main():
    control_path=ROOT/'.scratch/mcts-games-full-process.json'
    control=json.loads(control_path.read_text());pid=control['driver_pid']
    target=OUTPUT/'java-loveletter-4-shard0.json'
    gate=(OUTPUT/'java-tictactoe-2-shard1.stderr.log').resolve()
    assert gate.is_relative_to(OUTPUT.resolve())
    marker=gate/'schedule-change.txt'
    reason='User requested 250 games per setting. Stop before launching the next batch.\n'
    if gate.exists():
        assert gate.is_dir() and marker.read_text()==reason,'Unexpected existing gate/log; inspect before continuing'
    else:
        gate.mkdir();marker.write_text(reason)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        stamp('waiting_for_current_batch',original_driver_pid=pid,target_file=target.name,stop_gate=gate.name)
        while not target.exists():
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
            if code.value!=259:raise RuntimeError('Original collector stopped before committing its current batch; inspect logs')
            time.sleep(1)
        saved=json.loads(target.read_text())
        assert len(saved['games'])==100 and saved['identity']['games']==100
        stamp('waiting_for_original_collector_exit',original_driver_pid=pid)
        # The old collector opens the next stderr log before Popen. The owned
        # directory above makes it exit before launching another benchmark,
        # after its current child has exited and the completed batch is saved.
        deadline=time.monotonic()+60
        while True:
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
            if code.value!=259:break
            if time.monotonic()>deadline:raise RuntimeError('Old collector did not exit at the stop gate; inspect before resuming')
            time.sleep(.2)
        assert gate.is_dir() and marker.read_text()==reason
        marker.unlink();gate.rmdir()
        (ROOT/'.scratch/mcts-games-full-process-original.json').write_text(json.dumps(control,indent=2)+'\n')
    finally:kernel.CloseHandle(handle)
    command=[sys.executable,'-u',str(Path(__file__).with_name('run.py')),'--output',str(OUTPUT),
             '--games','250','--shards','3','--shard-sizes','100,100,50','--resize',
             '--iterations','5000','--warmup','6','--aligned','true','--seed-copies','true','--skip-forced','true','--report']
    with (ROOT/'.scratch/mcts-games-250-driver.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        control_path.write_text(json.dumps({'driver_pid':process.pid,'command':command,'log':'.scratch/mcts-games-250-driver.log'},indent=2)+'\n')
        plan=json.loads((OUTPUT/'collection-plan.json').read_text());plan['state']='running_revised_schedule'
        (OUTPUT/'collection-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        stamp('running_revised_schedule',driver_pid=process.pid)
        code=process.wait()
        stamp('complete' if code==0 else 'failed',driver_pid=process.pid,exit_code=code)
        sys.exit(code)

if __name__=='__main__':main()
