import core.*;
import core.actions.AbstractAction;
import games.tictactoe.*;
import games.connect4.*;
import games.cantstop.*;
import games.loveletter.*;
import players.mcts.*;
import players.PlayerConstants;
import java.util.*;
import org.json.simple.JSONObject;

/** Complete all-MCTS games over public TAG, with game-independent measurement hooks. */
public final class BenchmarkMctsGames {
    private static JSONObject object(Object... pairs){JSONObject o=new JSONObject();for(int i=0;i<pairs.length;i+=2)o.put(pairs[i],pairs[i+1]);return o;}
    private static AbstractGameState create(String game,int players,long seed,AbstractForwardModel model){
        AbstractParameters p=switch(game){case "tictactoe"->new TicTacToeGameParameters();case "connect4"->new Connect4GameParameters();
            case "cantstop"->new CantStopParameters();case "loveletter"->new LoveLetterParameters();default->throw new IllegalArgumentException(game);};
        p.setRandomSeed(seed);
        if(p instanceof LoveLetterParameters ll){ll.setParameterValue("nTokensWin2",6);ll._reset();}
        if(p instanceof CantStopParameters cs){
            String[] names={"TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE","TEN","ELEVEN","TWELVE"};
            int[] heights={3,5,7,9,11,13,11,9,7,5,3};
            for(int i=0;i<11;i++)cs.setParameterValue(names[i]+"_MAX",heights[i]);cs._reset();
        }
        long init=BenchmarkHooks.start(0);
        AbstractGameState state=switch(game){case "tictactoe"->new TicTacToeGameState(p,players);case "connect4"->new Connect4GameState(p,players);
            case "cantstop"->new CantStopGameState(p,players);case "loveletter"->new LoveLetterGameState(p,players);default->throw new IllegalArgumentException(game);};
        model.setup(state);BenchmarkHooks.end(0,init);return state;
    }
    private static MCTSPlayer agent(String game,int seat,long seed,int iterations,AbstractForwardModel model,AbstractGameState state,boolean aligned){
        MCTSParams p=new MCTSParams();p.setRandomSeed(seed);p.setParameterValue("budgetType",PlayerConstants.BUDGET_ITERATIONS);
        p.setParameterValue("budget",iterations);p.setParameterValue("rolloutLength",10);
        p.setParameterValue("information",game.equals("loveletter")?MCTSEnums.Information.Information_Set:MCTSEnums.Information.Open_Loop);
        p.setParameterValue("rolloutType",MCTSEnums.Strategies.RANDOM);p.setParameterValue("reuseTree",false);
        if(aligned){p.setParameterValue("K",Math.sqrt(2));p.setParameterValue("selectionPolicy",MCTSEnums.SelectionPolicy.ROBUST);p.setParameterValue("maxTreeDepth",1000);}
        // Updating tunable settings resets PlayerParameters' seed. Resolve it last.
        p.setRandomSeed(seed);
        MCTSPlayer agent=new MCTSPlayer(p);agent.setForwardModel(model);agent.setPlayerID(seat);agent.initializePlayer(state);return agent;
    }
    public static void main(String[] args)throws Exception{
        if(args.length!=10)throw new IllegalArgumentException("game players games iterations seed stride aligned seed_copies skip_forced warmup_decisions");
        String game=args[0];int players=Integer.parseInt(args[1]),games=Integer.parseInt(args[2]),iterations=Integer.parseInt(args[3]);
        long seed=Long.parseLong(args[4]);BenchmarkHooks.stride=Long.parseLong(args[5]);boolean aligned=Boolean.parseBoolean(args[6]);
        BenchmarkHooks.seedCopies=Boolean.parseBoolean(args[7]);boolean skipForced=Boolean.parseBoolean(args[8]);
        BenchmarkHooks.searchForced=!skipForced;int warmup=Integer.parseInt(args[9]);
        AbstractForwardModel model=switch(game){case "tictactoe"->new TicTacToeForwardModel();case "connect4"->new Connect4ForwardModel();
            case "cantstop"->new CantStopForwardModel();case "loveletter"->new LoveLetterForwardModel();default->throw new IllegalArgumentException(game);};
        BenchmarkHooks.reset(seed);
        for(int i=0;i<warmup;i++){
            var state=create(game,players,seed+i,model);int actor=state.getCurrentPlayer();
            while(model.computeAvailableActions(state).size()==1){model.next(state,model.computeAvailableActions(state).getFirst());actor=state.getCurrentPlayer();}
            var a=agent(game,actor,seed+100000+i,iterations,model,state,aligned);
            var observation=state.copy(actor);a.getAction(observation,model.computeAvailableActions(state));
        }
        long control=0;for(int i=0;i<10000;i++){long t=System.nanoTime();control+=System.nanoTime()-t;}
        System.gc();
        System.out.println(object("phase","ready","engine","java","game",game,"players",players,"timer_control_ns",control/10000.0));System.out.flush();
        Scanner input=new Scanner(System.in);input.nextLine();
        for(int id=0;id<games;id++){
            BenchmarkHooks.reset(seed+id);long clock=System.nanoTime();
            var state=create(game,players,seed+id,model);
            MCTSPlayer[] agents=new MCTSPlayer[players];for(int seat=0;seat<players;seat++)agents[seat]=agent(game,seat,(seed+id)*4L+100000+seat,iterations,model,state,aligned);
            int decisions=0,searchedDecisions=0;long searchNs=0,traceHash=1469598103934665603L;
            while(state.isNotTerminal()){
                int actor=state.getCurrentPlayer();var actions=model.computeAvailableActions(state);if(actions.isEmpty())throw new IllegalStateException("empty actions");
                var observation=state.copy(actor);AbstractAction action;
                if(skipForced&&actions.size()==1){action=actions.getFirst();}
                else{
                    BenchmarkHooks.scope=1;long start=System.nanoTime();action=agents[actor].getAction(observation,actions);
                    searchNs+=System.nanoTime()-start;BenchmarkHooks.scope=0;searchedDecisions++;
                    if(agents[actor].getRootVisits()!=iterations)throw new IllegalStateException("iteration budget not completed: "+agents[actor].getRootVisits());
                }
                if(!actions.contains(action))throw new IllegalStateException("illegal agent choice");
                for(char c:action.toString().toCharArray())traceHash=(traceHash^c)*1099511628211L;
                model.next(state,action);decisions++;if(decisions>=100000)throw new IllegalStateException("game failed to terminate");
            }
            long elapsed=System.nanoTime()-clock;List<Object> metrics=new ArrayList<>();
            for(int scope=0;scope<2;scope++)for(int op=0;op<BenchmarkHooks.OPS.length;op++){
                long[] r=BenchmarkHooks.data[scope][op];metrics.add(object("scope",scope==0?"real":"search","operation",BenchmarkHooks.OPS[op],"calls",r[0],"samples",r[1],"ns",r[2]));
            }
            System.out.println(object("phase","game","engine","java","game",game,"players",players,"id",id,"seed",seed+id,"decisions",decisions,
                "searched_decisions",searchedDecisions,"elapsed_ns",elapsed,"search_ns",searchNs,"probe_ns",BenchmarkHooks.probeNs,"trace_hash",Long.toUnsignedString(traceHash),"results",Arrays.stream(state.getPlayerResults()).map(Object::toString).toList(),"metrics",metrics));System.out.flush();
            input.nextLine();
        }
    }
}
