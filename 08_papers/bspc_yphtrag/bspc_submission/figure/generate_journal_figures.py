import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def box(ax, xy, w, h, title, lines, fc="#f8f9fb", ec="#4b5563", title_color="#111827"):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="top",
            fontsize=12, fontweight="bold", color=title_color)
    for i, line in enumerate(lines):
        ax.text(x + 0.04, y + h - 0.18 - i * 0.09, line, ha="left", va="top",
                fontsize=9.5, color="#374151")


def arrow(ax, start, end, color="#6b7280", lw=1.3):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
    ))


def make_motivation():
    fig, ax = plt.subplots(figsize=(9.54, 7.79))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "From Topical Relevance to Evidential Supportiveness",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111827")

    box(ax, (0.06, 0.44), 0.39, 0.38, "Relevance-oriented retrieval", [
        "Optimizes topical similarity",
        "Sensitive to terminology variation",
        "May rank non-supportive passages high",
    ], fc="#f9fafb", ec="#9ca3af")

    box(ax, (0.55, 0.44), 0.39, 0.38, "Supportiveness-aware reranking", [
        "Prioritizes answer-justifying evidence",
        "Reranks within a fixed candidate pool",
        "Requires reachable evidence from Stage I",
    ], fc="#f7fbff", ec="#2563eb")

    arrow(ax, (0.45, 0.63), (0.55, 0.63), color="#4b5563", lw=1.6)
    ax.text(0.50, 0.68, "SEV", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1d4ed8")

    box(ax, (0.12, 0.18), 0.27, 0.13, "Typical failure", [
        "Topically related",
        "Insufficient support",
    ], fc="#fff7ed", ec="#b45309", title_color="#7c2d12")

    box(ax, (0.61, 0.18), 0.27, 0.13, "Target behavior", [
        "Evidence can justify",
        "answer-level claims",
    ], fc="#f0fdf4", ec="#15803d", title_color="#14532d")

    arrow(ax, (0.255, 0.44), (0.255, 0.31), color="#9a3412")
    arrow(ax, (0.745, 0.44), (0.745, 0.31), color="#166534")

    fig.savefig("motivation.pdf", bbox_inches="tight")
    plt.close(fig)


def module(ax, xy, w, h, title, fc="#f8fafc", ec="#64748b"):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#111827")


def make_framework():
    fig, ax = plt.subplots(figsize=(13.14, 7.19))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "YpathRAG: Supportiveness-Aware Retrieval and Reranking",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111827")

    module(ax, (0.04, 0.48), 0.12, 0.12, "Question", fc="#f8fafc")
    module(ax, (0.23, 0.64), 0.18, 0.11, "Dense Retriever", fc="#eef2ff", ec="#4f46e5")
    module(ax, (0.23, 0.33), 0.18, 0.11, "Lexicon-guided\nSparse Retriever", fc="#ecfeff", ec="#0891b2")
    module(ax, (0.48, 0.48), 0.17, 0.12, "Score Normalization\n& Fusion", fc="#fefce8", ec="#a16207")
    module(ax, (0.70, 0.48), 0.16, 0.12, "Top-$k_1$\nCandidate Pool", fc="#f8fafc")
    module(ax, (0.70, 0.24), 0.16, 0.12, "SEV Dual-head\nPrediction", fc="#f0fdf4", ec="#15803d")
    module(ax, (0.48, 0.12), 0.17, 0.12, "Top-$k_2$\nSupportive Evidence", fc="#f0fdf4", ec="#15803d")
    module(ax, (0.23, 0.12), 0.18, 0.12, "LLM Answer\nGeneration", fc="#fdf2f8", ec="#be185d")

    arrow(ax, (0.16, 0.54), (0.23, 0.69))
    arrow(ax, (0.16, 0.54), (0.23, 0.38))
    arrow(ax, (0.41, 0.695), (0.48, 0.56), color="#4f46e5")
    arrow(ax, (0.41, 0.385), (0.48, 0.52), color="#0891b2")
    arrow(ax, (0.65, 0.54), (0.70, 0.54))
    arrow(ax, (0.78, 0.48), (0.78, 0.36), color="#15803d")
    arrow(ax, (0.70, 0.30), (0.65, 0.18), color="#15803d")
    arrow(ax, (0.48, 0.18), (0.41, 0.18), color="#be185d")

    ax.text(0.32, 0.81, "Stage I: candidate generation", ha="center", va="center",
            fontsize=10.5, color="#374151")
    ax.text(0.69, 0.39, "Stage II: supportiveness verification", ha="center", va="center",
            fontsize=10.5, color="#374151")

    fig.savefig("framework.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_motivation()
    make_framework()
