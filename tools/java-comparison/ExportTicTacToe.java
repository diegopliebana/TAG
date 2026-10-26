// Export reachable Tic-Tac-Toe positions and legal actions from the public Java engine.
// The saved output lets Rust tests compare behavior without running Java each time.
// Comparison tooling for public TAG commit d65ad79f04992e238663c37c812dc2c5f6ce7ff2.
// Calls TAG APIs. See docs/reference/source-notices.md.
import core.actions.AbstractAction;
import core.actions.SetGridValueAction;
import games.tictactoe.TicTacToeForwardModel;
import games.tictactoe.TicTacToeGameParameters;
import games.tictactoe.TicTacToeGameState;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringJoiner;

public final class ExportTicTacToe {
    private static final TicTacToeForwardModel MODEL = new TicTacToeForwardModel();
    private static final Set<String> VISITED = new HashSet<>();

    private static String signature(TicTacToeGameState state) {
        StringBuilder board = new StringBuilder();
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                board.append(state.getGridBoard().getElement(x, y).getComponentName());
            }
        }
        StringJoiner actions = new StringJoiner(",");
        for (AbstractAction action : MODEL.computeAvailableActions(state)) {
            SetGridValueAction move = (SetGridValueAction) action;
            actions.add(Integer.toString(move.getX() * 3 + move.getY()));
        }
        return String.join("|", board.toString(), Integer.toString(state.getCurrentPlayer()),
            Integer.toString(state.isNotTerminal() ? state.getCurrentPlayer() : -1),
            Integer.toString(state.getGameTick()), Integer.toString(state.getTurnCounter()),
            Integer.toString(state.getRoundCounter()), Integer.toString(state.getFirstPlayer()),
            state.getGameStatus().name(), state.getPlayerResults()[0].name(),
            state.getPlayerResults()[1].name(), Double.toString(state.getGameScore(0)),
            Double.toString(state.getGameScore(1)), actions.toString());
    }

    private static void explore(TicTacToeGameState state, String sequence, PrintWriter output) {
        String full = signature(state);
        if (!VISITED.add(full.substring(0, 9))) return;
        output.println(sequence + "\t" + full + "\t"
            + signature((TicTacToeGameState) state.copy(0)) + "\t"
            + signature((TicTacToeGameState) state.copy(1)));
        List<AbstractAction> actions = MODEL.computeAvailableActions(state);
        for (AbstractAction action : actions) {
            SetGridValueAction move = (SetGridValueAction) action;
            TicTacToeGameState child = (TicTacToeGameState) state.copy();
            MODEL.next(child, action);
            explore(child, sequence + (move.getX() * 3 + move.getY()), output);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) throw new IllegalArgumentException("Supply output TSV path");
        TicTacToeGameState state = new TicTacToeGameState(new TicTacToeGameParameters(), 2);
        MODEL.setup(state);
        try (PrintWriter output = new PrintWriter(Files.newBufferedWriter(Path.of(args[0]), StandardCharsets.UTF_8))) {
            output.println("# tag-java-tictactoe-v1 d65ad79f04992e238663c37c812dc2c5f6ce7ff2 clean-user-reported");
            explore(state, "", output);
            if (output.checkError()) throw new IllegalStateException("Could not write fixture");
        }
        if (VISITED.size() != 5478) throw new IllegalStateException("Unexpected position count: " + VISITED.size());
        System.out.println("Exported " + VISITED.size() + " Java positions");
    }
}
