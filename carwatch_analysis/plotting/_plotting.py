import contextlib
from typing import Optional, Sequence, Tuple, Dict, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from biopsykit.protocols.plotting import feature_boxplot, saliva_multi_feature_boxplot
from biopsykit.stats import StatsPipeline
from fau_colors import colors_all

from carwatch_analysis._types import path_t


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
        stats_kwargs = {"box_pairs": box_pairs, "pvalues": pvalues, "verbose": 0}

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
        stats_kwargs = {"box_pairs": box_pairs, "pvalues": pvalues, "verbose": False}

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


def multi_boxplot_sampling_delay(
    data: pd.DataFrame, order: Sequence[str], **kwargs
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    fig, axs = _plot_get_fig_ax_list(order, **kwargs)
    kwargs.pop("figsize", None)

    data_group = data.groupby("log_type")
    data.index = data.index.rename({"sample": "Sample"})

    for i, (key, ax) in enumerate(zip(order, axs)):
        df = data_group.get_group(key)
        feature_boxplot(data=df.reset_index(), x="Sample", y="time_diff_to_naive_min", ax=ax, **kwargs)
        ax.set_title(key)

        if i == 0:
            ax.set_ylabel(r"$\Delta s$ [min]")
        else:
            ax.set_ylabel(None)

    fig.tight_layout()
    return fig, axs


def multi_paired_plot_sampling_delay(
    data: pd.DataFrame, order: Sequence[str], **kwargs
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:

    fig, axs = _plot_get_fig_ax_list(order, **kwargs)
    kwargs.pop("figsize", None)

    data.index = data.index.rename({"sample": "Sample"})
    data_group = data.groupby("log_type")

    for key, ax in zip(order, axs):
        df = data_group.get_group(key)
        _plot_paired(
            data=df.reset_index(),
            dv="time_diff_to_naive_min",
            within="Sample",
            subject="night_id",
            pointplot_kwargs={"alpha": 0.5},
            boxplot_in_front=True,
            ax=ax,
            **kwargs,
        )
        ax.yaxis.set_minor_locator(mticks.AutoMinorLocator())
        ax.set_title(key, fontsize="smaller")

    axs[0].set_ylabel(r"$\Delta s$ [min]")

    fig.tight_layout()
    return fig, axs


def multi_paired_plot_auc(
    data: pd.DataFrame, saliva_feature: str, log_types: Sequence[str], **kwargs
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    fig, axs = _plot_get_fig_ax_list(log_types, **kwargs)

    data = data.xs(saliva_feature, level="saliva_feature")

    title_dict = {"auc_g": "$AUC_G$", "auc_i": "$AUC_I$"}

    for log_type, ax in zip(log_types[::-1], axs):
        order = log_types.copy()
        order.remove(log_type)

        data_plot = data.reindex(order, level="log_type")

        _plot_paired(
            data=data_plot.reset_index(),
            dv="cortisol",
            within="log_type",
            order=order,
            subject="night_id",
            boxplot_in_front=True,
            pointplot_kwargs={"alpha": 0.5},
            ax=ax,
        )
        ax.yaxis.set_minor_locator(mticks.AutoMinorLocator())
        ax.set_xlabel("Log Type")
        ax.set_ylabel(r"Cortisol AUC $\left[\frac{nmol \cdot min}{l} \right]$")

    fig.suptitle(title_dict[saliva_feature])
    fig.tight_layout()
    return fig, axs


def paired_plot_auc(
    data: pd.DataFrame, saliva_feature: str, log_types: Sequence[str], **kwargs
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    fig, ax = _plot_get_fig_ax(**kwargs)

    data = data.xs(saliva_feature, level="saliva_feature")
    data = data.reindex(log_types, level="log_type")

    title_dict = {"auc_g": "$AUC_G$", "auc_i": "$AUC_I$"}

    pg.plot_paired(
        data=data.reset_index(),
        dv="cortisol",
        within="log_type",
        order=log_types,
        subject="night_id",
        boxplot_in_front=True,
        pointplot_kwargs={"alpha": 0.5},
        ax=ax,
    )
    ax.set_xlabel("Log Type")
    ax.set_ylabel(r"Cortisol AUC $\left[\frac{nmol \cdot min}{l} \right]$")

    fig.suptitle(title_dict[saliva_feature])
    fig.tight_layout()
    return fig, ax


def time_unit_digits_histogram_grid(
    data: pd.DataFrame, x: str, condition_order: Sequence[str], log_type_order: Sequence[str], **kwargs
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    fig = plt.figure(figsize=kwargs.get("figsize"), constrained_layout=True)
    fig.suptitle(kwargs.get("suptitle", None))

    grouper_condition = data.groupby("condition")
    subfigs = fig.subfigures(nrows=len(grouper_condition), ncols=1, hspace=0.05)

    for condition, subfig in zip(condition_order, subfigs):
        subfig.suptitle(condition, fontsize="medium")
        grouper_log_type = grouper_condition.get_group(condition).groupby("log_type")

        # create subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=len(grouper_log_type), gridspec_kw={"wspace": 0.1})
        for log_type, ax in zip(log_type_order, axs):
            df = grouper_log_type.get_group(log_type)
            sns.histplot(
                data=df.reset_index(),
                x=x,
                stat="percent",
                bins=10,
                binrange=[0, 9],
                discrete=True,
                ax=ax,
            )
            ax.set_xticks(np.arange(0, 10))
            ax.yaxis.set_major_locator(mticks.MultipleLocator(20))
            ax.yaxis.set_minor_locator(mticks.MultipleLocator(10))
            ax.set_title(log_type)
            ax.set_xlabel("Unit Digit [min]")
            ax.set_ylabel("Frequency [%]")
            ax.set_ylim(kwargs.get("ylim", None))
    return fig, axs


def time_unit_digits_histogram(
    data: pd.DataFrame, x: str, log_type_order: Sequence[str], **kwargs
) -> Tuple[plt.Figure, plt.Axes]:

    fig = plt.figure(figsize=kwargs.get("figsize"), constrained_layout=True)
    fig.suptitle(kwargs.get("suptitle", None))

    subfig = fig.subfigures(nrows=1, ncols=1, hspace=0.05)
    subfig.suptitle("All Conditions", fontsize="medium")

    grouper = data.groupby("log_type")

    axs = subfig.subplots(nrows=1, ncols=len(grouper), gridspec_kw={"wspace": 0.1})

    for log_type, ax in zip(log_type_order, axs):
        df = grouper.get_group(log_type)

        sns.histplot(data=df.reset_index(), x=x, stat="percent", bins=10, binrange=[0, 9], discrete=True, ax=ax)
        ax.set_xticks(np.arange(0, 10))
        ax.yaxis.set_major_locator(mticks.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mticks.MultipleLocator(10))
        ax.set_title(log_type)
        ax.set_xlabel("Unit Digit [min]")
        ax.set_ylabel("Frequency [%]")
        ax.set_ylim(kwargs.get("ylim"))

    return fig, axs


def _plot_paired(
    data: pd.DataFrame,
    dv: Optional[str] = None,
    within: Optional[str] = None,
    subject: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    boxplot: Optional[bool] = True,
    boxplot_in_front: Optional[bool] = False,
    orient: Optional[str] = "v",
    figsize: Optional[Tuple[int]] = (4, 4),
    dpi: Optional[int] = 100,
    ax: Optional[plt.Axes] = None,
    colors: Optional[Sequence[str]] = None,
    pointplot_kwargs: Optional[Dict[str, Any]] = None,
    boxplot_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Paired plot.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Long-format dataFrame.
    dv : string
        Name of column containing the dependent variable.
    within : string
        Name of column containing the within-subject factor.
    subject : string
        Name of column containing the subject identifier.
    order : list of str
        List of values in ``within`` that define the order of elements on the
        x-axis of the plot. If None, uses alphabetical order.
    boxplot : boolean
        If True, add a boxplot to the paired lines using the
        :func:`seaborn.boxplot` function.
    boxplot_in_front : boolean
        If True, the boxplot is plotted on the foreground (i.e. above the
        individual lines) and with a slight transparency. This makes the
        overall plot more readable when plotting a large numbers of subjects.
    orient : string
        Plot the boxplots vertically and the subjects on the x-axis if
        ``orient='v'`` (default). Set to ``orient='h'`` to rotate the plot by 90 degrees.
    figsize : tuple
        Figsize in inches
    dpi : int
        Resolution of the figure in dots per inches.
    ax : matplotlib axes
        Axis on which to draw the plot.
    colors : list of str
        Line colors names. Default is green when value increases from A to B,
        indianred when value decreases from A to B and grey when the value is
        the same in both measurements.
    pointplot_kwargs : dict
        Dictionnary of optional arguments that are passed to the
        :py:func:`seaborn.pointplot` function.
    boxplot_kwargs : dict
        Dictionnary of optional arguments that are passed to the
        func:`seaborn.boxplot` function.

    Returns
    -------
    ax : :class:`matplotlib.Axes`
        Returns the Axes object with the plot for further tweaking.

    Notes
    -----
    Data must be a long-format pandas DataFrame. Missing values are automatically removed using a
    strict listwise approach (= complete-case analysis).

    Examples
    --------
    Default paired plot:

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset("mixed_anova").query("Time != 'January'")
        >>> df = df.query("Group == 'Meditation' and Subject > 40")
        >>> ax = pg.plot_paired(data=df, dv="Scores", within="Time",
        ...                     subject="Subject", dpi=150)

    Paired plot on an existing axis (no boxplot and uniform color):

    .. plot::

        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> df = pg.read_dataset("mixed_anova").query("Time != 'January'")
        >>> df = df.query("Group == 'Meditation' and Subject > 40")
        >>> fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
        >>> pg.plot_paired(data=df, dv="Scores", within="Time",
        ...                subject="Subject", ax=ax1, boxplot=False,
        ...                colors=["grey", "grey", "grey"])  # doctest: +SKIP

    Horizontal paired plot with three unique within-levels:

    .. plot::

        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> df = pg.read_dataset("mixed_anova").query("Group == 'Meditation'")
        >>> # df = df.query("Group == 'Meditation' and Subject > 40")
        >>> pg.plot_paired(data=df, dv="Scores", within="Time",
        ...                subject="Subject", orient="h")  # doctest: +SKIP

    With the boxplot on the foreground:

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset("mixed_anova").query("Time != 'January'")
        >>> df = df.query("Group == 'Control'")
        >>> ax = pg.plot_paired(data=df, dv="Scores", within="Time",
        ...                     subject="Subject", boxplot_in_front=True)
    """
    if boxplot_kwargs is None:
        boxplot_kwargs = {"color": "lightslategrey", "width": 0.2}
    if pointplot_kwargs is None:
        pointplot_kwargs = {"scale": 0.6, "marker": "."}
    if colors is None:
        colors = ["green", "grey", "indianred"]
    from pingouin.utils import _check_dataframe

    # Update default kwargs with specified inputs
    _pointplot_kwargs = {"scale": 0.6, "marker": "."}
    _pointplot_kwargs.update(pointplot_kwargs)
    _boxplot_kwargs = {"color": "lightslategrey", "width": 0.2}
    _boxplot_kwargs.update(boxplot_kwargs)
    # Extract pointplot alpha, if set
    pp_alpha = _pointplot_kwargs.pop("alpha", 1.0)

    # Calculate size of the plot elements by scale as in Seaborn pointplot
    scale = _pointplot_kwargs.pop("scale")
    lw = plt.rcParams["lines.linewidth"] * 1.8 * scale  # get the linewidth
    mew = lw * 0.75  # get the markeredgewidth
    markersize = np.pi * np.square(lw) * 2  # get the markersize

    # Set boxplot in front of Line2D plot (zorder=2 for both) and add alpha
    if boxplot_in_front:
        _boxplot_kwargs.update(
            {
                "boxprops": {"zorder": 2},
                "whiskerprops": {"zorder": 2},
                "zorder": 2,
            }
        )

    # Validate args
    data = _check_dataframe(data=data, dv=dv, within=within, subject=subject, effects="within")

    # Pivot and melt the table. This has several effects:
    # 1) Force missing values to be explicit (a NaN cell is created)
    # 2) Automatic collapsing to the mean if multiple within factors are present
    # 3) If using dropna, remove rows with missing values (listwise deletion).
    # The latter is the same behavior as JASP (= strict complete-case analysis).
    data_piv = data.pivot_table(index=subject, columns=within, values=dv, observed=True)
    data_piv = data_piv.dropna()
    data = data_piv.melt(ignore_index=False, value_name=dv).reset_index()

    # Extract within-subject level (alphabetical order)
    x_cat = np.unique(data[within])

    if order is None:
        order = x_cat
    else:
        assert len(order) == len(
            x_cat
        ), "Order must have the same number of elements as the number of levels in `within`."

    # Substitue within by integer order of the ordered columns to allow for
    # changing the order of numeric withins.
    data["wthn"] = data[within].replace({_ordr: i for i, _ordr in enumerate(order)})
    order_num = range(len(order))  # Make numeric order

    # Start the plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Set x and y depending on orientation using the num. replacement within
    _x = "wthn" if orient == "v" else dv
    _y = dv if orient == "v" else "wthn"

    for cat in range(len(x_cat) - 1):
        _order = (order_num[cat], order_num[cat + 1])
        # Extract data of the current subject-combination
        data_now = data.loc[data["wthn"].isin(_order), [dv, "wthn", subject]]
        # Select colors for all lines between the current subjects
        y1 = data_now.loc[data_now["wthn"] == _order[0], dv].to_numpy()
        y2 = data_now.loc[data_now["wthn"] == _order[1], dv].to_numpy()
        # Line and scatter colors depending on subject dv trend
        _colors = np.where(y1 < y2, colors[0], np.where(y1 > y2, colors[2], colors[1]))
        # Line and scatter colors as hue-indexed dictionary
        _colors = {subj: clr for subj, clr in zip(data_now[subject].unique(), _colors)}
        # Plot individual lines using Seaborn
        sns.lineplot(
            data=data_now,
            x=_x,
            y=_y,
            hue=subject,
            palette=_colors,
            ls="-",
            lw=lw,
            legend=False,
            ax=ax,
        )
        # Plot individual markers using Seaborn
        sns.scatterplot(
            data=data_now,
            x=_x,
            y=_y,
            hue=subject,
            palette=_colors,
            edgecolor="face",
            lw=mew,
            sizes=[markersize] * data_now.shape[0],
            legend=False,
            ax=ax,
            **_pointplot_kwargs,
        )

    # Set zorder and alpha of pointplot markers and lines
    _ = plt.setp(ax.collections, alpha=pp_alpha, zorder=2)  # Set marker alpha
    _ = plt.setp(ax.lines, alpha=pp_alpha, zorder=2)  # Set line alpha

    if boxplot:
        # Set boxplot x and y depending on orientation
        _xbp = within if orient == "v" else dv
        _ybp = dv if orient == "v" else within
        sns.boxplot(data=data, x=_xbp, y=_ybp, order=order, ax=ax, orient=orient, **_boxplot_kwargs)

        # Set alpha to patch of boxplot but not to whiskers
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.75))
    else:
        # If no boxplot, axis needs manual styling as in Seaborn pointplot
        if orient == "v":
            xlabel, ylabel = within, dv
            ax.set_xticks(np.arange(len(x_cat)))
            ax.set_xticklabels(order)
            ax.xaxis.grid(False)
            ax.set_xlim(-0.5, len(x_cat) - 0.5, auto=None)
        else:
            xlabel, ylabel = dv, within
            ax.set_yticks(np.arange(len(x_cat)))
            ax.set_yticklabels(order)
            ax.yaxis.grid(False)
            ax.set_ylim(-0.5, len(x_cat) - 0.5, auto=None)
            ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Despine
    sns.despine(ax=ax)
    return ax


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


def _plot_get_fig_ax(**kwargs):
    ax: plt.Axes = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize"))
    else:
        fig = ax.get_figure()
    return fig, ax


def _plot_get_fig_ax_list(order: Sequence[str], **kwargs) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    axs: Sequence[plt.Axes] = kwargs.get("axs", None)
    if axs is None:
        fig, axs = plt.subplots(figsize=kwargs.get("figsize"), ncols=len(order), sharey=True)
    else:
        fig = axs[0].get_figure()
    return fig, axs
