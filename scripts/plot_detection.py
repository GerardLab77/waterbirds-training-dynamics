"""Plot the recorded per-seed detector comparison, without training."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["scope"] == "all"]
    seeds = sorted({int(r["seed"]) for r in rows})
    methods = ("mean_loss", "class_relative_loss")
    if len(rows) != len(seeds)*len(methods):
        raise ValueError("Expected one global row per seed and detector")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":11})
    fig, axes = plt.subplots(1,2,figsize=(9,4.3),constrained_layout=True)
    for ax, metric, title in zip(axes,("precision_at_10pct","average_precision"),
                                ("Precision among the highest-scored 10%","Average precision")):
        for j,(method,label,color) in enumerate(zip(methods,("Mean loss","Class-relative loss"),("#0072B2","#D55E00"))):
            values = [float(next(r[metric] for r in rows if int(r["seed"])==seed and r["detector"]==method)) for seed in seeds]
            positions = [i+(j-.5)*.34 for i in range(len(seeds))]
            bars = ax.bar(positions,values,width=.31,color=color,label=label)
            ax.bar_label(bars,labels=[f"{100*v:.1f}%" if metric.startswith("precision") else f"{v:.3f}" for v in values],padding=3,fontsize=10)
        prevalence = float(rows[0]["prevalence"])
        ax.axhline(prevalence,color="#666666",linestyle="--",linewidth=1,label="Dataset prevalence")
        ax.set(title=title,xlabel="Model seed",xticks=list(range(len(seeds))),xticklabels=[str(s) for s in seeds],ylim=(0,.5))
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
        if metric.startswith("precision"):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="upper left",fontsize=9,frameon=False)
    fig.suptitle("Waterbirds: class-relative scoring improves the early-loss detector",fontsize=13)
    fig.supxlabel("First five epochs · same 4,795 training images across seeds · exploratory comparison",fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
