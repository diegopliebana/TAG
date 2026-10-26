// Research-only timing/retention harness over unmodified public TAG APIs.
// No game implementation is copied here. Build output belongs in the disposable workspace.
import core.*;
import core.actions.AbstractAction;
import games.tictactoe.*;
import games.connect4.*;
import games.cantstop.*;
import games.loveletter.*;
import players.simple.*;
import players.mcts.*;
import players.PlayerConstants;
import java.util.*;
import java.lang.management.ManagementFactory;

public final class BenchmarkResearch {
    private static volatile Object sink;
    private static volatile long checksum;
    private final String game;
    private final int players;
    private final long seed;
    private final AbstractForwardModel model;
    private final AbstractParameters parameters;
    private final AbstractGameState root;
    private final List<AbstractAction> actions;
    private final int preparedActions;
    private final Random random;
    private final RandomPlayer randomPlayer;
    private final MCTSPlayer mcts;
    private final com.sun.management.ThreadMXBean allocation;
    private long gameOrdinal;
    // games, accepted actions, logical turns including terminal, completed rounds, observations
    private final long[] gameWork = new long[5];
    private BenchmarkResearch(String game,int players,long seed,int iterations,int prefix,boolean aligned) {
        this.game=game;this.players=players;this.seed=seed;
        switch(game) {
            case "tictactoe": model=new TicTacToeForwardModel();parameters=new TicTacToeGameParameters();break;
            case "connect4": model=new Connect4ForwardModel();parameters=new Connect4GameParameters();break;
            case "cantstop": model=new CantStopForwardModel();parameters=new CantStopParameters();break;
            case "loveletter": model=new LoveLetterForwardModel();parameters=new LoveLetterParameters();break;
            default:throw new IllegalArgumentException(game);
        }
        parameters.setRandomSeed(seed);
        if(aligned && parameters instanceof LoveLetterParameters p){p.setParameterValue("nTokensWin2",6);p._reset();}
        var prepared=create();if(game.equals("cantstop"))model.next(prepared,model.computeAvailableActions(prepared).getFirst());
        var preparationRandom=new Random(seed+91);int advanced=0;
        for(int i=0;i<prefix;i++){
            var legal=model.computeAvailableActions(prepared);var previous=prepared.copy();
            model.next(prepared,legal.get(preparationRandom.nextInt(legal.size())));
            if(!prepared.isNotTerminal()){prepared=previous;break;}advanced++;
        }
        root=prepared;preparedActions=advanced;
        actions=model.computeAvailableActions(root);require(!actions.isEmpty(),"nonempty root actions");
        random=new Random(seed);
        randomPlayer=new RandomPlayer(new Random(seed));randomPlayer.setForwardModel(model);randomPlayer.setPlayerID(root.getCurrentPlayer());
        MCTSParams p=new MCTSParams();p.setRandomSeed(seed);
        p.setParameterValue("budgetType",PlayerConstants.BUDGET_ITERATIONS);
        p.setParameterValue("budget",iterations);p.setParameterValue("rolloutLength",4);
        p.setParameterValue("information",game.equals("loveletter")?MCTSEnums.Information.Information_Set:MCTSEnums.Information.Open_Loop);
        mcts=new MCTSPlayer(p);mcts.setForwardModel(model);mcts.setPlayerID(root.getCurrentPlayer());mcts.initializePlayer(root);
        allocation=(com.sun.management.ThreadMXBean)ManagementFactory.getThreadMXBean();
        allocation.setThreadAllocatedMemoryEnabled(true);
    }
    private static void require(boolean ok,String message){if(!ok)throw new IllegalStateException(message);}
    private AbstractGameState create(){
        AbstractGameState state=switch(game){
            case "tictactoe" -> new TicTacToeGameState(parameters,players);
            case "connect4" -> new Connect4GameState(parameters,players);
            case "cantstop" -> new CantStopGameState(parameters,players);
            case "loveletter" -> new LoveLetterGameState(parameters,players);
            default -> throw new IllegalArgumentException(game);
        };
        model.setup(state);return state;
    }
    private long bytes(){return allocation.getThreadAllocatedBytes(Thread.currentThread().threadId());}
    private void operation(String op){
        switch(op){
            case "control":sink=this;break;
            case "init":sink=create();break;
            case "copy":sink=root.copy();break;
            case "observe":sink=root.copy(root.getCurrentPlayer());break;
            case "actions":sink=model.computeAvailableActions(root);checksum+=((List<?>)sink).size();break;
            case "validate":sink=model.computeAvailableActions(root).contains(actions.getFirst());break;
            case "history_copy":sink=List.of(root.getHistory(),root.getHistoryAsText());break;
            case "transition":{var state=root.copy();model.next(state,actions.getFirst());sink=state;break;}
            case "query":{checksum+=root.getCurrentPlayer();sink=root.getPlayerResults()[root.getCurrentPlayer()];checksum+=(long)root.getGameScore(root.getCurrentPlayer());break;}
            case "fork":sink=root.copy(root.getCurrentPlayer());break;
            case "first":sink=actions.getFirst();break;
            case "random":sink=randomPlayer.getAction(root,actions);break;
            case "osla":{var agent=new OSLAPlayer(new Random(seed));agent.setForwardModel(model);agent.setPlayerID(root.getCurrentPlayer());sink=agent.getAction(root.copy(root.getCurrentPlayer()),actions);break;}
            case "mcts":sink=mcts.getAction(root.copy(root.getCurrentPlayer()),actions);break;
            case "game": case "game_direct": case "game_aligned": case "game_aligned_direct":{
                boolean observe=op.equals("game")||op.equals("game_aligned");
                var state=create();int steps=0;
                var choices=new Random(seed+(gameOrdinal++%32));
                var agent=new RandomPlayer(choices);agent.setForwardModel(model);
                int turns=0;
                while(state.isNotTerminal()){
                    var moves=model.computeAvailableActions(state);require(!moves.isEmpty(),"game choices");
                    AbstractAction chosen;
                    if(observe){
                        int actor=state.getCurrentPlayer();agent.setPlayerID(actor);
                        var observation=state.copy(actor);
                        chosen=agent.getAction(observation,moves);
                    }else chosen=moves.get(moves.size()==1?0:choices.nextInt(moves.size()));
                    if(!game.equals("cantstop") || chosen instanceof games.cantstop.actions.Pass)turns++;
                    model.next(state,chosen);steps++;require(steps<100_000,"termination bound");
                }
                gameWork[0]++;gameWork[1]+=steps;gameWork[2]+=turns;
                gameWork[3]+=state.getRoundCounter();gameWork[4]+=observe?steps:0;
                checksum+=steps;sink=state;break;
            }
            default:throw new IllegalArgumentException(op);
        }
    }
    private static long heap(){return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();}
    private void retain(int count)throws Exception{
        Scanner input=new Scanner(System.in);
        System.gc();Thread.sleep(100);
        long baseline=heap();System.out.printf("{\"phase\":\"idle\",\"heap_used_bytes\":%d}%n",baseline);System.out.flush();
        input.nextLine();long before=bytes();
        var states=new ArrayList<AbstractGameState>();for(int i=0;i<count;i++)states.add(root.copy());
        long allocated=bytes()-before;sink=states;System.gc();Thread.sleep(100);
        System.out.printf("{\"phase\":\"loaded\",\"count\":%d,\"heap_used_bytes\":%d,\"allocated_bytes\":%d}%n",count,heap(),allocated);System.out.flush();input.nextLine();sink=states;
    }
    private void benchmark(String op,int count,int warmMs,int sampleMs){
        if(op.equals("apply") || op.equals("apply_control")){
            long warm=System.nanoTime();do{applyBlock(op);}while(System.nanoTime()-warm<warmMs*1_000_000L);
            for(int batch=0;batch<5;batch++){
                long calls=0,elapsed=0,allocated=0;
                do{long[] sample=applyBlock(op);calls+=32;elapsed+=sample[0];allocated+=sample[1];}while(calls<count||elapsed<sampleMs*1_000_000L);
                System.out.printf(Locale.ROOT,"{\"batch\":%d,\"calls\":%d,\"elapsed_ns\":%d,\"allocated_bytes\":%d}%n",batch,calls,elapsed,allocated);
            }
            System.out.printf("{\"prepared_actions\":%d,\"root_actions\":%d,\"seed\":%d}%n",preparedActions,actions.size(),seed);return;
        }
        long start=System.nanoTime();do{operation(op);}while(System.nanoTime()-start<warmMs*1_000_000L);
        for(int batch=0;batch<5;batch++){
            long[] beforeWork=gameWork.clone();
            long before=bytes();start=System.nanoTime();long calls=0;
            do{for(int block=0;block<32;block++){operation(op);calls++;}}while(calls<count||System.nanoTime()-start<sampleMs*1_000_000L);
            long elapsed=System.nanoTime()-start;long allocated=bytes()-before;
            long[] measuredWork=new long[5];for(int i=0;i<5;i++)measuredWork[i]=gameWork[i]-beforeWork[i];
            System.out.printf(Locale.ROOT,"{\"batch\":%d,\"calls\":%d,\"elapsed_ns\":%d,\"allocated_bytes\":%d,\"game_work\":%s}%n",batch,calls,elapsed,allocated,Arrays.toString(measuredWork));
        }
        if(op.equals("mcts"))require(mcts.getRootVisits()==mcts.getParameters().budget,"MCTS iteration count");
        System.out.printf("{\"checksum\":%d,\"root_actions\":%d,\"seed\":%d,\"search_visits\":%d,\"prepared_actions\":%d,\"history_entries\":%d}%n",checksum,actions.size(),seed,op.equals("mcts")?mcts.getRootVisits():0,preparedActions,root.getHistory().size());
    }
    private long[] applyBlock(String op){
        var states=new AbstractGameState[32];for(int i=0;i<32;i++)states[i]=root.copy();
        long before=bytes(),start=System.nanoTime();
        if(op.equals("apply"))for(var state:states)model.next(state,actions.getFirst());
        long elapsed=System.nanoTime()-start,allocated=bytes()-before;sink=states;
        return new long[]{elapsed,allocated};
    }
    public static void main(String[] args)throws Exception{
        require(args.length==8,"game players operation count seed iterations warm_ms sample_ms");
        String[] operation=args[2].split("@",2);int prefix=operation.length==2?Integer.parseInt(operation[1]):0;
        var bench=new BenchmarkResearch(args[0],Integer.parseInt(args[1]),Long.parseLong(args[4]),Integer.parseInt(args[5]),prefix,operation[0].startsWith("game_aligned"));
        if(operation[0].equals("retain_states"))bench.retain(Integer.parseInt(args[3]));
        else bench.benchmark(operation[0],Integer.parseInt(args[3]),Integer.parseInt(args[6]),Integer.parseInt(args[7]));
    }
}
