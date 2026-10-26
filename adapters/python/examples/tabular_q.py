"""Run the educational Q-learning example; reusable code lives in learning.tabular."""
from tag_rust.learning import tabular as _implementation
# Retain historical example imports, including teaching helpers.
globals().update({k:v for k,v in vars(_implementation).items() if not k.startswith("__")})

if __name__ == "__main__":
    main()
