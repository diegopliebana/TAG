// Export Connect 4 move sequences and resulting boards/actions from public Java TAG.
// A signature is the fixed-format summary of one position used by Rust comparison tests.
// Calls the recorded public TAG APIs from the disposable build copy.
import core.actions.AbstractAction;
import core.actions.SetGridValueAction;
import games.connect4.Connect4ForwardModel;
import games.connect4.Connect4GameParameters;
import games.connect4.Connect4GameState;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.StringJoiner;

public final class ExportConnect4 {
    private static final Connect4ForwardModel MODEL = new Connect4ForwardModel();
    private static String signature(Connect4GameState state) {
        StringBuilder board = new StringBuilder();
        for (int x=0;x<state.getGridBoard().getWidth();x++)
            for (int y=0;y<state.getGridBoard().getHeight();y++)
                board.append(state.getGridBoard().getElement(x,y).getComponentName());
        StringJoiner actions = new StringJoiner(",");
        for (AbstractAction action: MODEL.computeAvailableActions(state)) {
            SetGridValueAction move=(SetGridValueAction)action;
            actions.add(move.getX()+":"+move.getY());
        }
        StringJoiner winning = new StringJoiner(",");
        for (var cell:state.getWinningCells()) winning.add(cell.a+":"+cell.b);
        return String.join("|",board.toString(),Integer.toString(state.getCurrentPlayer()),
            Integer.toString(state.isNotTerminal()?state.getCurrentPlayer():-1),
            Integer.toString(state.getGameTick()),Integer.toString(state.getTurnCounter()),
            Integer.toString(state.getRoundCounter()),Integer.toString(state.getFirstPlayer()),
            state.getGameStatus().name(),state.getPlayerResults()[0].name(),state.getPlayerResults()[1].name(),
            Double.toString(state.getGameScore(0)),Double.toString(state.getGameScore(1)),actions.toString(),winning.toString());
    }
    public static void main(String[] args) throws Exception {
        if(args.length!=1) throw new IllegalArgumentException("Supply output TSV path");
        int records=0;
        try(PrintWriter output=new PrintWriter(Files.newBufferedWriter(Path.of(args[0]),StandardCharsets.UTF_8))) {
            output.println("# tag-java-connect4-v1 d65ad79f04992e238663c37c812dc2c5f6ce7ff2 clean-user-reported");
            for(int[] config:new int[][]{{7,6,4},{5,4,3},{10,8,6},{3,2,3}}) {
                for(int trial=0;trial<32;trial++) {
                    Connect4GameParameters p=new Connect4GameParameters();
                    p.setParameterValue("width",config[0]);p.setParameterValue("height",config[1]);p.setParameterValue("winCount",config[2]);
                    Connect4GameState state=new Connect4GameState(p,2); MODEL.setup(state);
                    Random random=new Random(1000+trial);
                    StringJoiner history=new StringJoiner(",");
                    while(true) {
                        output.println(config[0]+","+config[1]+","+config[2]+"\t"+history+"\t"+signature(state)+"\t"
                            +signature((Connect4GameState)state.copy(0))+"\t"+signature((Connect4GameState)state.copy(1)));
                        records++;
                        if(!state.isNotTerminal()) break;
                        List<AbstractAction> actions=MODEL.computeAvailableActions(state);
                        // Include deterministic first/last-choice games and seeded varied legal play.
                        int index=trial==0?0:trial==1?actions.size()-1:random.nextInt(actions.size());
                        SetGridValueAction move=(SetGridValueAction)actions.get(index);
                        history.add(Integer.toString(move.getX())); MODEL.next(state,move);
                    }
                }
            }
            if(output.checkError()) throw new IllegalStateException("Could not write fixture");
        }
        System.out.println("Exported "+records+" prefixes across 128 games");
    }
}
