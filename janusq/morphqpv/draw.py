def plot_results():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    data = pd.read_csv(f"{respath}time.csv")
    datamean = data.groupby(["qubits"]).mean().reset_index().sort_values(["qubits"])
    datamean.reset_index(drop=True, inplace=True)
    scale = 0.009
    fig = plt.figure(figsize=(1000 * scale, 900 * scale))
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.size"] = 62
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    # set axes linewidth
    mpl.rcParams["axes.linewidth"] = 5
    ## set ticks linewidth
    mpl.rcParams["xtick.major.size"] = 20
    mpl.rcParams["xtick.major.width"] = 5
    axes = plt.axes([0, 0, 1, 1])
    colors = ["#7CBEEC", "#7EB57E", "#FF6F45", "#3274A1"]
    names = [
        "classical simulation",
        "state tomography",
        "process tomography",
        "our methods",
    ]
    datamean = (
        data[data.qubits.isin([6, 8, 10])]
        .groupby(["qubits"])
        .mean()
        .reset_index()
        .sort_values(["qubits"])
    )
    for i, col in enumerate(
        [
            "simulate_time",
            "total_state_tomography_time",
            "total_process_tomography_time",
            "infer_time",
        ]
    ):
        axes.bar(
            datamean.index + i * 0.2,
            datamean[col],
            color=colors[i],
            label=names[i],
            linewidth=1,
            width=0.2,
        )
    axes.set_yscale("log")
    axes.tick_params(axis="x", which="major", width=5, length=20)
    axes.set_yticks([1, 100, 10000, 1000000])
    # axes.set_xticklabels(allnames,rotation=90)
    axes.set_xlabel("# qubits")
    axes.set_xticks(datamean.index + 0.3)
    axes.set_xticklabels(datamean.qubits)
    axes.grid(axis="y", color="gray", linestyle="--", linewidth=5)
    axes.set_ylabel("time(s)")

    axes.legend(frameon=False, bbox_to_anchor=(1, 1.4), ncol=2)
    fig.savefig(f"{respath}fig9(a).svg", dpi=600, format="svg", bbox_inches="tight")

