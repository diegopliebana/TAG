"""Reserve a figure-wide band for legends, outside every plotting area."""
import math


def layout_with_legend(fig):
    entries = {}
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        for handle, label in zip(legend.legend_handles, legend.get_texts()):
            entries.setdefault(label.get_text(), handle)
        legend.remove()
    if not entries:
        fig.tight_layout()
        return
    columns = min(4, len(entries))
    rows = math.ceil(len(entries) / columns)
    # Reserve physical space so both short and tall figures keep a clear gap.
    band_inches = .28 * rows + .22
    top = 1 - band_inches / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(list(entries.values()), list(entries), loc="upper center",
               bbox_to_anchor=(.5, 1), ncol=columns, frameon=False, fontsize=9)
