// Scripted choices are applied by Rust to produce the positions used in the preview.
const limits = { max_cells: 1000000 };
const marks = coordinates => coordinates.map(([x, y]) => ({ x, y }));
const columns = values => values.map(column => ({ column }));

export const scenes = [
  { id: "ttt-win", title: "Tic-Tac-Toe / standard win", game: "tictactoe", parameters: { grid_size: 3, limits },
    moves: marks([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]), outcome: "won",
    emphasis: [[0, 0], [1, 0], [2, 0]] },
  { id: "ttt-draw", title: "Tic-Tac-Toe / draw", game: "tictactoe", parameters: { grid_size: 3, limits },
    moves: marks([[0, 0], [1, 1], [2, 2], [0, 2], [0, 1], [2, 1], [1, 2], [1, 0], [2, 0]]), outcome: "draw" },
  { id: "ttt-large", title: "Tic-Tac-Toe / 5 x 5", game: "tictactoe", parameters: { grid_size: 5, limits },
    moves: marks([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4]]), outcome: "won",
    emphasis: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]] },
  { id: "ttt-small", title: "Tic-Tac-Toe / 1 x 1", game: "tictactoe", parameters: { grid_size: 1, limits },
    moves: marks([[0, 0]]), outcome: "won", emphasis: [[0, 0]] },
  { id: "c4-win", title: "Connect 4 / standard win", game: "connect4", parameters: { width: 7, height: 6, winning_length: 4, limits },
    moves: columns([0, 1, 0, 1, 0, 1, 0]), outcome: "won" },
  { id: "c4-configured", title: "Connect 4 / 4 x 3, target 3", game: "connect4", parameters: { width: 4, height: 3, winning_length: 3, limits },
    moves: columns([0, 1, 0, 1, 0]), outcome: "won" },
  { id: "c4-draw", title: "Connect 4 / draw", game: "connect4", parameters: { width: 4, height: 3, winning_length: 4, limits },
    moves: columns([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]), outcome: "draw" },
];

// A request-capable client supplies either the real browser worker or the test Wasm engine.
export async function buildSample(client, scene) {
  let reply = await client.request({ type: "create", game: scene.game, parameters: scene.parameters }, { replace: true });
  const views = [reply.view];
  for (const action of scene.moves) {
    reply = await client.request({ type: "human_move", actor: reply.view.next_player, action }, { positionVersion: reply.view.position_version });
    views.push(reply.view);
  }
  if (reply.view.board.outcome.kind !== scene.outcome) throw new Error(`Unexpected sample outcome: ${scene.id}`);
  // These coordinates annotate the scripted Tic-Tac-Toe examples for presentation review.
  if (scene.emphasis) {
    const winner = reply.view.board.outcome.player;
    for (const [x, y] of scene.emphasis) {
      if (reply.view.board.cells[x * reply.view.board.height + y] !== winner) throw new Error(`Invalid sample emphasis: ${scene.id}`);
    }
  }
  return { scene, views };
}
