package core;

import java.util.*;
import core.actions.AbstractAction;

/** Measurement hooks compiled only into the disposable benchmark classes. */
public final class BenchmarkHooks {
    public static final String[] OPS={"init","actions","copy","observe","copy_apply","apply","fork","actual_copy"};
    public static long[][][] data=new long[2][8][3]; // actual calls, timed samples, ns
    public static int scope=0;
    public static long stride=1024, probeNs=0, seed=42, copyOrdinal=0;
    public static boolean paused=false,seedCopies=false,searchForced=false;
    public static volatile Object sink;
    public static Random copyRandom(){return seedCopies?new Random(seed+(paused?0:++copyOrdinal)*6364136223846793005L):new Random();}
    public static void reset(long gameSeed){data=new long[2][8][3];probeNs=0;scope=0;seed=gameSeed;copyOrdinal=0;}
    public static long start(int op){
        if(paused||stride==0)return 0;
        long count=++data[scope][op][0];
        return scope==0||count<=16||count%stride==0?System.nanoTime():0;
    }
    public static void end(int op,long start){
        if(start==0)return;
        long elapsed=System.nanoTime()-start;data[scope][op][1]++;data[scope][op][2]+=elapsed;
    }
    public static long nextStart(AbstractForwardModel model,AbstractGameState state,AbstractAction action){
        if(paused||stride==0)return 0;
        long count=data[scope][5][0]+1;
        if(scope==0||count<=16||count%stride==0){
            long before=System.nanoTime();Random saved=state.redeterminisationRnd;
            paused=true;state.redeterminisationRnd=new Random(0);
            try{
                AbstractAction independent=action.copy();
                long t=System.nanoTime();AbstractGameState branch=state.copy();long copyNs=System.nanoTime()-t;
                model.next(branch,independent);long pairNs=System.nanoTime()-t;
                sink=branch;
                data[scope][2][1]++;data[scope][2][2]+=copyNs;
                data[scope][4][1]++;data[scope][4][2]+=pairNs;
            } finally {sink=null;state.redeterminisationRnd=saved;paused=false;probeNs+=System.nanoTime()-before;}
        }
        return start(5);
    }
}
