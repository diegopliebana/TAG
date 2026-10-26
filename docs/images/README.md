# Explanatory figures

These SVG images are part of the teaching documentation, separate from the
publisher artwork used by the game UI. They illustrate existing encodings; they
illustrate the array layouts described in the text. Producing image observations
for learning remains a separate implementation task.

- learning-board-vector.svg: board coordinates and column-by-column vector order.
- learning-board-tensor.svg: the same board split into own/opposing/empty channels.

Both show Player 0 observing X at (1,2) and O at (2,0), with zero-based coordinates.
Labels and numbers carry meaning as well as color. Each SVG contains a title and
text description, and its Markdown use supplies alternative text.

To update the figures, edit their source and regenerate with no extra packages:

```powershell
python tools/docs/generate-learning-diagrams.py
```

Keep the figures consistent with the Rust encoder and adjacent text. Commit the
generated SVG files alongside source changes so readers can view the Markdown
without running a tool. SVG remains sharp when enlarged; readers can open an image
directly if their Markdown viewer displays it too small.
