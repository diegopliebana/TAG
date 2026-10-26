"""Play both games from Python; Rust owns their boards and applies all rules."""
import tag_rust


def play(name, parameters=None):
    with tag_rust.create(name, parameters) as game:
        spec = game.spec()
        while game.next_player is not None:
            view = game.observe(game.next_player, payloads=("board",))
            # IDs arrive in the game's offered-action order. Choose the first here.
            action_id = int(view.offered_action_ids[0])
            game.apply(action_id, position=view.position)
        final = game.observe(0, payloads=())
        print(f"{spec.identity.game}: {final.progress[0]} moves, results={final.results.tolist()}")


if __name__ == "__main__":
    play("tictactoe")
    play("connect4")
    play("connect4", {"width": 4, "height": 3, "winning_length": 3})
