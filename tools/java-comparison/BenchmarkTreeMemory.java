// Retained MCTS-tree memory on the unmodified public Java framework.
import core.*;
import games.GameType;
import players.mcts.*;
import players.PlayerConstants;
import java.util.*;
import java.lang.management.ManagementFactory;
public final class BenchmarkTreeMemory {
    private static volatile Object sink;
    public static void main(String[] a)throws Exception {
        if(a.length!=3)throw new IllegalArgumentException("game players iterations");
        String name=switch(a[0]){case "tictactoe"->"TicTacToe";case "connect4"->"Connect4";case "cantstop"->"CantStop";case "loveletter"->"LoveLetter";default->throw new IllegalArgumentException(a[0]);};
        int players=Integer.parseInt(a[1]),iterations=Integer.parseInt(a[2]);
        Game game=GameType.valueOf(name).createGameInstance(players,42L);var fm=game.getForwardModel();var state=game.getGameState();
        if(a[0].equals("cantstop"))fm.next(state,fm.computeAvailableActions(state).getFirst());
        int forced=0;
        while(fm.computeAvailableActions(state).size()==1){fm.next(state,fm.computeAvailableActions(state).getFirst());if(++forced>100)throw new IllegalStateException("No branching fixture");}
        var observation=state.copy(state.getCurrentPlayer());var actions=fm.computeAvailableActions(observation);
        if(actions.size()<2)throw new IllegalStateException("Branching fixture required");
        MCTSParams p=new MCTSParams();p.setRandomSeed(42L);p.setParameterValue("budgetType",PlayerConstants.BUDGET_ITERATIONS);p.setParameterValue("budget",iterations);p.setParameterValue("rolloutLength",4);p.setParameterValue("information",a[0].equals("loveletter")?MCTSEnums.Information.Information_Set:MCTSEnums.Information.Open_Loop);
        MCTSPlayer agent=new MCTSPlayer(p);agent.setForwardModel(fm);agent.setPlayerID(state.getCurrentPlayer());agent.initializePlayer(observation);
        long start=System.nanoTime();do{agent.getAction(observation,actions);}while(System.nanoTime()-start<1_000_000_000L);
        agent.initializePlayer(observation); // Release the warmup tree before the baseline.
        Scanner input=new Scanner(System.in);var bean=(com.sun.management.ThreadMXBean)ManagementFactory.getThreadMXBean();bean.setThreadAllocatedMemoryEnabled(true);
        System.gc();Thread.sleep(100);long baseline=ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
        System.out.printf("{\"phase\":\"idle\",\"heap_used_bytes\":%d}%n",baseline);System.out.flush();input.nextLine();
        long allocated=bean.getThreadAllocatedBytes(Thread.currentThread().threadId());agent.getAction(observation,actions);allocated=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())-allocated;
        if(agent.getRootVisits()!=iterations)throw new IllegalStateException("iteration count mismatch");sink=agent.getRoot();System.gc();Thread.sleep(100);
        System.out.printf("{\"phase\":\"loaded\",\"iterations\":%d,\"visits\":%d,\"allocated_bytes\":%d,\"heap_used_bytes\":%d}%n",iterations,agent.getRootVisits(),allocated,ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed());System.out.flush();input.nextLine();sink=agent.getRoot();
    }
}
