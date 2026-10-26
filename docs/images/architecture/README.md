# Architecture figures

Each figure shows a focused view of the implemented system. Cards identify the
main components and add short notes about their responsibilities or data. Boxes group
responsibilities; they are not a class hierarchy. Arrows show use or data flow.
In the three sequence diagrams, time runs downward and dashed lines identify
participants. Shaded groups and role colors separate responsibilities; labels and
arrows remain meaningful without color.

| View and explanation | Standalone exports |
|---|---|
| [Overall system](../../reference/framework/overview.md#modules-and-execution-boundaries) | [SVG](system.svg), [PDF](system.pdf) |
| [One native decision](../../reference/agents/implementing-agents.md#one-native-decision) | [SVG](native-decision.svg), [PDF](native-decision.pdf) |
| [Actual state and possible worlds](../../reference/framework/contracts.md#observations-and-hypothetical-states) | [SVG](information.svg), [PDF](information.pdf) |
| [Neural evaluation](../../reference/learning/neural-mcts.md#advance-evaluate-and-resume) | [SVG](neural-evaluation.svg), [PDF](neural-evaluation.pdf) |
| [Browser move](../../reference/browser/README.md#a-browser-move) | [SVG](browser-move.svg), [PDF](browser-move.pdf) |
| [Experiments](../../reference/experiments/README.md#from-settings-to-results) | [SVG](experiments.svg), [PDF](experiments.pdf) |
| [Saved artifacts](../../reference/persistence/README.md#choose-the-file-for-the-task) | [SVG](artifacts.svg), [PDF](artifacts.pdf) |

## Reproduce or edit the figures

Both formats come from one [Python source](../../../tools/docs/generate-architecture-diagrams.py).
It uses only Python's standard library. From the repository root:

```powershell
python tools/docs/generate-architecture-diagrams.py
python tools/docs/generate-architecture-diagrams.py --check
```

The first command rewrites the exports; `--check` only checks that they match the
source and exits with an error if an export is missing or outdated. On a POSIX
shell, use `python3` if that is your Python executable. The script also works
from another directory when invoked with its full path.

SVGs include an accessible title and description; the linked reference text
explains each flow. PDFs are single vector pages for reuse in documents or
slides. They can be enlarged without pixelation; no screenshot or external
renderer is needed to generate them.
