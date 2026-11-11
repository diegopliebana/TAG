"""Compile hooks in an isolated class directory; never modify the public reference."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'references/tag-public'
BUILD=ROOT/'.scratch/mcts-games'

def wrap(source,signature,start,end):
    pos=source.index(signature);opening=source.index('{',pos);depth=1;i=opening+1
    # These selected method bodies contain no braces in character/string literals.
    while depth:
        depth+=(source[i]=='{')-(source[i]=='}');i+=1
    return source[:opening+1]+start+source[opening+1:i-1]+end+source[i-1:]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--java-build',type=Path,default=ROOT/'.scratch/java-comparison')
    args=parser.parse_args();reference=args.reference.resolve()
    BUILD.mkdir(parents=True,exist_ok=True)
    base=args.java_build.resolve()
    if not (base/'target/comparison-classpath.txt').exists():raise RuntimeError('Build the disposable Java reference as documented in tools/benchmarks/research/README.md')
    dependencies=(base/'target/comparison-classpath.txt').read_text().strip().split(';')
    # The game/core probe needs no networking/native-platform JARs. Keep the
    # compile classpath focused; the existing full classpath remains for execution.
    selected=[p for p in dependencies if any(s in p.lower() for s in ('json-simple','commons-','guava','failureaccess','gson'))]
    cp=str(base/'target/classes')+';'+';'.join(selected)
    src=BUILD/'src/core';src.mkdir(parents=True,exist_ok=True)
    original=(reference/'src/main/java/core/AbstractGameState.java').read_text()
    state=original.replace('protected Random redeterminisationRnd = new Random();','protected Random redeterminisationRnd = BenchmarkHooks.copyRandom();')
    assert state!=original
    state=wrap(state,'public final AbstractGameState copy(int playerId)',
        '\nlong benchmarkStart=BenchmarkHooks.start(playerId == -1 ? 7 : 3);try {\n',
        '\n} finally {BenchmarkHooks.end(playerId == -1 ? 7 : 3,benchmarkStart);}\n')
    (src/'AbstractGameState.java').write_text(state)
    model=(reference/'src/main/java/core/AbstractForwardModel.java').read_text()
    model=wrap(model,'public final void next(AbstractGameState currentState, AbstractAction action)',
        '\nlong benchmarkStart=BenchmarkHooks.nextStart(this,currentState,action);try {\n',
        '\n} finally {BenchmarkHooks.end(5,benchmarkStart);}\n')
    model=wrap(model,'public final List<AbstractAction> computeAvailableActions(AbstractGameState gameState, ActionSpace actionSpace, int activePlayer)',
        '\nlong benchmarkStart=BenchmarkHooks.start(1);try {\n',
        '\n} finally {BenchmarkHooks.end(1,benchmarkStart);}\n')
    (src/'AbstractForwardModel.java').write_text(model)
    player=(reference/'src/main/java/players/mcts/MCTSPlayer.java').read_text()
    patched=player.replace('if (actions.size() == 1)','if (!core.BenchmarkHooks.searchForced && actions.size() == 1)')
    assert player!=patched
    player_source=BUILD/'src/players/mcts/MCTSPlayer.java';player_source.parent.mkdir(parents=True,exist_ok=True);player_source.write_text(patched)
    loveletter=(reference/'src/main/java/games/loveletter/LoveLetterGameState.java').read_text()
    patched=loveletter.replace('Random redeterminisationRnd = new Random(System.currentTimeMillis());','Random redeterminisationRnd = core.BenchmarkHooks.copyRandom();')
    assert loveletter!=patched
    loveletter_source=BUILD/'src/games/loveletter/LoveLetterGameState.java';loveletter_source.parent.mkdir(parents=True,exist_ok=True);loveletter_source.write_text(patched)
    classes=BUILD/'classes';classes.mkdir(exist_ok=True)
    subprocess.run(['javac','-proc:none','-implicit:none','-sourcepath',str(BUILD/'empty-source'),'-cp',cp,'-d',str(classes),str(src/'AbstractGameState.java'),str(src/'AbstractForwardModel.java'),str(player_source),str(loveletter_source),str(HERE/'BenchmarkHooks.java'),str(HERE/'BenchmarkMctsGames.java')],check=True)
    (BUILD/'classpath.txt').write_text(str(classes)+';'+str(base/'target/classes')+';'+';'.join(selected))
    subprocess.run(['cargo','build','--offline','--release','--features','mcts-json','--example','mcts_games_measure'],cwd=ROOT,check=True)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'rust_executable':digest(ROOT/'target/release/examples/mcts_games_measure.exe'),
              'classes':{str(p.relative_to(classes)):digest(p) for p in sorted(classes.rglob('*.class'))},
              'tool_sources':{p.name:digest(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.rs','.java','.py')},
              'java_reference':{str(p.relative_to(reference)):digest(p) for folder in ('core','players/mcts','games/tictactoe','games/connect4','games/cantstop','games/loveletter') for p in sorted((reference/'src/main/java'/folder).rglob('*.java'))}}
    (BUILD/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
