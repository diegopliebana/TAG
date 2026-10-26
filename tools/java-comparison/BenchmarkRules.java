// Time the same recorded moves and public rule operations as the Rust measurement tool.
// Setup, validation and warmup are separate from the timed batches; see the benchmark report.
// Matched public-API benchmarks for the recorded public TAG build.
import core.AbstractForwardModel;
import core.AbstractGameState;
import core.AbstractParameters;
import core.actions.AbstractAction;
import core.actions.SetGridValueAction;
import games.tictactoe.TicTacToeForwardModel;
import games.tictactoe.TicTacToeGameParameters;
import games.tictactoe.TicTacToeGameState;
import games.connect4.Connect4ForwardModel;
import games.connect4.Connect4GameParameters;
import games.connect4.Connect4GameState;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class BenchmarkRules {
    private static volatile Object sink;
    private static volatile long checksum;
    private static final String[] MODES = {"init", "actions", "copy", "transition", "replay"};
    private static final int BATCHES = 5;
    private record Replay(int[] moves, String board, String result) {}
    private record Frame(AbstractGameState state, AbstractAction action) {}
    private record Sample(long elapsed, long operations) {}
    private final String game;
    private final AbstractForwardModel model;
    private final AbstractParameters parameters;
    private final List<Replay> replays;
    private final List<Frame> frames = new ArrayList<>();
    private long actionCount;

    private BenchmarkRules(String game, List<String> corpus) {
        this.game = game;
        if (game.equals("ttt")) {
            model = new TicTacToeForwardModel();
            parameters = new TicTacToeGameParameters();
        } else {
            model = new Connect4ForwardModel();
            parameters = new Connect4GameParameters();
        }
        parameters.setRandomSeed(123);
        replays = new ArrayList<>();
        for (String line : corpus) {
            if (!line.startsWith(game + "\t")) continue;
            String[] fields = line.split("\t");
            String expected = game.equals("ttt") ? "3,3,3" : "7,6,4";
            require(String.join(",", fields[1], fields[2], fields[3]).equals(expected), "configuration");
            String[] codes = fields[4].split(",");
            int[] moves = new int[codes.length];
            for (int i = 0; i < codes.length; i++) moves[i] = Integer.parseInt(codes[i]);
            replays.add(new Replay(moves, fields[5], fields[6]));
        }
        require(replays.size() == 32, "32 recorded games required");
        prepare();
    }
    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalStateException(message);
    }
    private AbstractGameState create() {
        AbstractGameState state = game.equals("ttt")
            ? new TicTacToeGameState(parameters, 2)
            : new Connect4GameState(parameters, 2);
        model.setup(state);
        return state;
    }
    private AbstractAction selected(List<AbstractAction> actions, int code) {
        for (AbstractAction action : actions) {
            SetGridValueAction move = (SetGridValueAction) action;
            int identity = game.equals("ttt") ? move.getX() * 3 + move.getY() : move.getX();
            if (identity == code) return action;
        }
        throw new IllegalStateException("recorded move unavailable");
    }
    private String board(AbstractGameState state) {
        var grid = game.equals("ttt") ? ((TicTacToeGameState)state).getGridBoard()
                                     : ((Connect4GameState)state).getGridBoard();
        StringBuilder result = new StringBuilder();
        for (int x = 0; x < grid.getWidth(); x++)
            for (int y = 0; y < grid.getHeight(); y++)
                result.append(grid.getElement(x, y).getComponentName());
        return result.toString();
    }
    private void prepare() {
        for (Replay replay : replays) {
            AbstractGameState state = create();
            for (int code : replay.moves()) {
                require(state.isNotTerminal(), "prefix must be ongoing");
                List<AbstractAction> actions = model.computeAvailableActions(state);
                actionCount += actions.size();
                AbstractAction action = selected(actions, code);
                frames.add(new Frame(state.copy(), action));
                model.next(state, action);
            }
            require(!state.isNotTerminal(), "terminal replay required");
            require(board(state).equals(replay.board()), "terminal board mismatch");
            require(state.getPlayerResults()[0].name().equals(replay.result()), "terminal result mismatch");
        }
        System.out.printf("corpus\t%s\tgames=%d\tprefixes=%d\tavailable_actions=%d%n",
            game, replays.size(), frames.size(), actionCount);
        System.out.printf("settings\t%s\tcompetitionMode=%s\tseed=123%n", game, create().getCoreGameParameters().competitionMode);
    }
    private Sample run(String mode, int passes) {
        long elapsed = 0;
        long observed = 0;
        for (int pass = 0; pass < passes; pass++) {
            // Prepare independent prefix copies before timing transition application.
            List<AbstractGameState> states = new ArrayList<>();
            if (mode.equals("transition"))
                for (Frame frame : frames) states.add(frame.state().copy());
            long start = System.nanoTime();
            switch (mode) {
                case "init":
                    for (Replay ignored : replays) sink = create();
                    break;
                case "actions":
                    for (Frame frame : frames) {
                        List<AbstractAction> actions = model.computeAvailableActions(frame.state());
                        observed += actions.size();
                        sink = actions;
                    }
                    break;
                case "copy":
                    for (Frame frame : frames) sink = frame.state().copy();
                    break;
                case "transition":
                    for (int i = 0; i < frames.size(); i++) {
                        AbstractGameState state = states.get(i);
                        model.next(state, frames.get(i).action());
                        sink = state;
                    }
                    break;
                case "replay":
                    for (Replay replay : replays) {
                        AbstractGameState state = create();
                        for (int code : replay.moves()) {
                            List<AbstractAction> actions = model.computeAvailableActions(state);
                            model.next(state, selected(actions, code));
                        }
                        sink = state;
                    }
                    break;
                default: throw new IllegalArgumentException(mode);
            }
            elapsed += System.nanoTime() - start;
        }
        if (mode.equals("actions")) require(observed == (long)passes * actionCount, "action-count mismatch");
        checksum = observed;
        int operations = mode.equals("init") || mode.equals("replay") ? replays.size() : frames.size();
        return new Sample(elapsed, (long)passes * operations);
    }
    private void benchmark(int passes) {
        for (String mode : MODES) {
            long start = System.nanoTime();
            int warmed = 0;
            while (System.nanoTime() - start < 1_000_000_000L || warmed < 200) {
                run(mode, 10);
                warmed += 10;
            }
            System.out.printf("warmup\t%s\t%s\tpasses=%d%n", game, mode, warmed);
            for (int batch = 1; batch <= BATCHES; batch++) {
                Sample result = run(mode, passes);
                System.out.printf(Locale.ROOT, "sample\tjava\t%s\t%s\t%d\t%d\t%.3f%n",
                    game, mode, batch, result.operations(), (double)result.elapsed() / result.operations());
            }
        }
    }
    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("Supply corpus path and passes per batch");
        int passes = Integer.parseInt(args[1]);
        if (passes < 1 || passes > 10000) throw new IllegalArgumentException("passes must be 1..10000");
        System.out.printf("java\tversion=%s\tvm=%s\tpasses=%d%n", System.getProperty("java.version"), System.getProperty("java.vm.name"), passes);
        List<String> corpus = Files.readAllLines(Path.of(args[0]));
        for (String game : new String[]{"ttt", "c4"}) new BenchmarkRules(game, corpus).benchmark(passes);
    }
}
