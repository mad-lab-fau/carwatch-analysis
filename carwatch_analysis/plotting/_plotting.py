import contextlib
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import pandas as pd
import seaborn as sns
from biopsykit.protocols.plotting import feature_boxplot, saliva_multi_feature_boxplot
from biopsykit.stats import StatsPipeline
from fau_colors import colors_all

from carwatch_analysis._types import path_t


def boxplot_imu_features(
    data: pd.DataFrame,
    hue: str,
    feature: str,
    pipeline: Optional[StatsPipeline] = None,
):

    ylabels = {
        "sm_max_60": "Duration [s]",
        "sm_number_60": "Number",
        "sm_max_position": "Time before Awak. [min]",
        "sm_mean_60": "Duration [s]",
        "sm_std_60": "Duration [s]",
    }

    xticklabels = {
        "sm_max_60": r"$sm_{max}$",
        "sm_number_60": "|sm|",
        "sm_max_position": r"$t(sm_{max})$",
        "sm_mean_60": r"$\mu_{sm}$",
        "sm_std_60": r"$\sigma_{sm}$",
    }

    data_plot = data.xs(feature, level="imu_feature", drop_level=False)

    stats_kwargs = None
    if pipeline is not None:
        box_pairs, pvalues = pipeline.sig_brackets(
            "posthoc",
            stats_effect_type="between",
            plot_type="multi",
            features=[feature],
            x="imu_feature",
        )
        stats_kwargs = {"box_pairs": box_pairs, "pvalues": pvalues}

    with contextlib.redirect_stdout(None):
        fig, ax = feature_boxplot(
            data=data_plot,
            x="imu_feature",
            y="data",
            hue=hue,
            ylabel=ylabels[feature],
            xticklabels=[xticklabels[feature]],
            stats_kwargs=stats_kwargs,
            legend_fontsize="small",
            legend_orientation="horizontal",
            legend_loc="upper center",
        )
        ax.set_xlabel(None)


def boxplot_saliva_features(
    data: pd.DataFrame,
    hue: str,
    pipeline: Optional[StatsPipeline] = None,
    export: Optional[bool] = True,
    export_path: Optional[path_t] = None,
    **kwargs,
):
    features = {"auc": ["auc_g", "auc_i"], "slope": ["slopeS0S3", "slopeS0S4"], "max_inc": ["max_inc"]}

    stats_kwargs = None
    if pipeline is not None:
        box_pairs, pvalues = pipeline.sig_brackets(
            "posthoc",
            stats_effect_type="between",
            plot_type="multi",
            x="saliva_feature",
            features=features,
            subplots=True,
        )
        stats_kwargs = {"box_pairs": box_pairs, "pvalues": pvalues}

    with contextlib.redirect_stdout(None):
        fig, axs = saliva_multi_feature_boxplot(
            data=data,
            saliva_type="cortisol",
            features=features,
            hue=hue,
            stats_kwargs=stats_kwargs,
            legend_fontsize="small",
            legend_orientation="horizontal",
            legend_loc="upper center",
            **kwargs,
        )

    if export:
        fig.savefig(export_path.joinpath(f"img_boxplots_car_features_{hue}.pdf"), transparent=True)


def static_moment_plot(
    data: pd.DataFrame,
    wake_onset: pd.Timestamp,
    static_moments: Optional[pd.DataFrame] = None,
    plot_static_moment_features: Optional[bool] = False,
    **kwargs,
):
    if kwargs.get("ax", None):
        ax = kwargs.get("ax")
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=(7, 5))

    sns.set_palette(kwargs.get("palette", sns.light_palette(getattr(colors_all, "fau"), n_colors=4, reverse=True)[:-1]))

    data.filter(like="acc").plot(ax=ax)
    # Wake Onset vline
    ax.vlines(
        [wake_onset],
        0,
        1,
        transform=ax.get_xaxis_transform(),
        linewidth=3,
        linestyles="--",
        colors=colors_all.nat,
        zorder=3,
    )

    # Wake Onset Text + Arrow
    ax.annotate(
        "Wake Onset",
        xy=(mdates.date2num(wake_onset), 0.90),
        xycoords=ax.get_xaxis_transform(),
        xytext=(mdates.date2num(wake_onset - pd.Timedelta("3min")), 0.90),
        textcoords=ax.get_xaxis_transform(),
        ha="right",
        va="center",
        bbox=dict(
            fc=(1, 1, 1, plt.rcParams["legend.framealpha"]),
            ec=plt.rcParams["legend.edgecolor"],
            boxstyle="round",
        ),
        size=14,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color=colors_all.nat,
            shrinkA=0.0,
            shrinkB=0.0,
        ),
    )

    ax.set_title(None)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mticks.AutoMinorLocator(6))
    ax.set_ylabel("Acceleration $[m/s]$")
    ax.set_xlabel("Time")

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout(pad=0.2)
    ax.legend(loc="lower left")

    if static_moments is not None:
        _plot_static_moments(static_moments, plot_static_moment_features, wake_onset, ax)

    return fig, ax


def _plot_static_moments(
    static_moments: pd.DataFrame, plot_static_moment_features: bool, wake_onset: pd.Timestamp, ax: plt.Axes
) -> plt.Axes:
    handle = None

    sm_dur = static_moments.diff(axis=1)["end"].dt.total_seconds()
    color = colors_all.tech_dark

    for i, row in static_moments.iterrows():
        # add/subtract 10 seconds for better visibility
        td = pd.Timedelta("10s")
        handle = ax.axvspan(row["start"] + td, row["end"] - td, color=colors_all.tech, alpha=0.1)

        if plot_static_moment_features:
            ax.annotate(
                "",
                xy=(mdates.date2num(row["start"]), 0.60),
                xytext=(mdates.date2num(row["end"]), 0.60),
                xycoords=ax.get_xaxis_transform(),
                arrowprops=dict(arrowstyle="<->", color=color, lw=2, shrinkA=0.0, shrinkB=0.0),
            )

            ax.annotate(
                f"{sm_dur.loc[i]:.0f} s",
                xy=(
                    mdates.date2num(row["start"]) + 0.5 * (mdates.date2num(row["end"]) - mdates.date2num(row["start"])),
                    0.60,
                ),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0, 10),
                textcoords="offset points",
                color=color,
                ha="center",
                fontsize="x-small",
                bbox=dict(fc=(1, 1, 1, plt.rcParams["legend.framealpha"]), ec=None),
            )

            if i == 4:
                ax.annotate(
                    "$sp_{max}$",
                    xy=(
                        mdates.date2num(row["start"])
                        + 0.5 * (mdates.date2num(row["end"]) - mdates.date2num(row["start"])),
                        0.70,
                    ),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(0, 10),
                    textcoords="offset points",
                    color=color,
                    ha="center",
                    fontsize="small",
                    bbox=dict(
                        fc=(1, 1, 1, plt.rcParams["legend.framealpha"]),
                        ec=plt.rcParams["legend.edgecolor"],
                        boxstyle="round",
                    ),
                )

                ax.annotate(
                    "",
                    xy=(mdates.date2num(row["start"]), 0.20),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(mdates.date2num(wake_onset), 0.20),
                    color=color,
                    ha="center",
                    fontsize="small",
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkA=0.0, shrinkB=0.0),
                )
                ax.annotate(
                    "$t(sp_{max})$",
                    xy=(
                        mdates.date2num(row["start"])
                        + 0.5 * (mdates.date2num(wake_onset) - mdates.date2num(row["start"])),
                        0.20,
                    ),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -20),
                    textcoords="offset points",
                    color=color,
                    ha="center",
                    fontsize="small",
                    bbox=dict(
                        fc=(1, 1, 1, plt.rcParams["legend.framealpha"]),
                        ec=plt.rcParams["legend.edgecolor"],
                        boxstyle="round",
                    ),
                )

            if i == 0:
                ax.annotate(
                    "",
                    xy=(
                        mdates.date2num(row["start"])
                        + 0.5 * (mdates.date2num(wake_onset) - mdates.date2num(row["start"])),
                        0.40,
                    ),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -10),
                    textcoords="offset points",
                    color=color,
                    ha="center",
                    fontsize="small",
                    arrowprops=dict(
                        arrowstyle="-[, widthB=13.0, lengthB=0.5", color=color, lw=2, shrinkA=0.0, shrinkB=0.0
                    ),
                )
                ax.annotate(
                    r"$|sp|$, $\mu_{sp}$, $\sigma_{sp}$",
                    xy=(
                        mdates.date2num(row["start"])
                        + 0.5 * (mdates.date2num(wake_onset) - mdates.date2num(row["start"])),
                        0.40,
                    ),
                    xytext=(0, -30),
                    textcoords="offset points",
                    xycoords=ax.get_xaxis_transform(),
                    color=color,
                    ha="center",
                    fontsize="small",
                    bbox=dict(
                        fc=(1, 1, 1, plt.rcParams["legend.framealpha"]),
                        ec=plt.rcParams["legend.edgecolor"],
                        boxstyle="round",
                    ),
                )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(handle)
    labels.append("Static Moments")
    ax.legend(handles, labels, loc="lower left", fontsize="small")

    return ax
