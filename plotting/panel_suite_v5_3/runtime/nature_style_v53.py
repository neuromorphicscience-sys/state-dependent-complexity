"""Unified V5.3 style contract for the Neural Science figure suite.

Goals for V5.3
--------------
1. Match the cleaner look of the user's preferred reference panels:
   - filled markers instead of hollow circles
   - restrained ribbons / confidence bands
   - softer but higher-contrast palette
   - smaller, cleaner annotation text
2. Keep *all* computational and biological outputs; this package only changes
   rendering style and physical axis normalization.
3. Generate leaf panels for manual assembly, not final composite figures.
"""
from __future__ import annotations

import json, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pandas as pd

MM_PER_INCH = 25.4
PANEL_HEIGHT_MM = float(os.environ.get("NATURE_PANEL_HEIGHT_MM", "64"))
OUTPUT_DPI = int(os.environ.get("NATURE_OUTPUT_DPI", "600"))
FONT_FAMILY = "Arial"

# Slightly smaller and cleaner than V5.2 to avoid heavy text crowding.
FONT_SIZE_BASE = 6.2
FONT_SIZE_LABEL = 6.4
FONT_SIZE_TICK = 5.4
FONT_SIZE_LEGEND = 5.4
FONT_SIZE_ANNOT = 5.8
FONT_SIZE_TITLE_MAX = 6.8
FONT_SIZE_PANEL = 8.0

LINEWIDTH_AXIS = 0.55
LINEWIDTH_MAIN = 1.00
LINEWIDTH_DATA = 0.95
LINEWIDTH_SECONDARY = 0.75
LINEWIDTH_REFERENCE = 0.75
TICK_LENGTH = 2.5
MARKER_SIZE = 3.8
MARKER_SIZE_SMALL = 3.0
ALPHA_POINTS = 0.88
ALPHA_BAND = 0.12
ALPHA_FILL = 0.12

COLORS = {
    # preferred reference-like colors
    "teal": "#2A9D8F",
    "blue": "#38598C",
    "purple": "#8B7DAA",
    "orange": "#C1743C",
    "olive": "#90A86B",
    "rose": "#C97C7C",
    "slate": "#6B8FA8",
    "black": "#111111",
    "dark_gray": "#4D4D4D",
    "mid_gray": "#8C8C8C",
    "light_gray": "#CFCFCF",
    "very_light_gray": "#EEEEEE",
    "white": "#FFFFFF",
    # semantic aliases
    "complexity": "#38598C",
    "leverage": "#2A9D8F",
    "biology": "#C1743C",
    "support": "#8B7DAA",
    "secondary": "#6B8FA8",
    "accent": "#90A86B",
    "mismatch": "#B86B4B",
    "same_state": "#2A9D8F",
    "cross_state": "#B86B4B",
    "null": "#8C8C8C",
    "random": "#8C8C8C",
    "baseline": "#4D4D4D",
    "shuffle": "#CFCFCF",
}

STATE_COLORS = {
    "sparse": COLORS["teal"],
    "sparse drive": COLORS["teal"],
    "transition mid": COLORS["purple"],
    "transition-mid": COLORS["purple"],
    "transition dense": COLORS["blue"],
    "transition-dense": COLORS["blue"],
}
DATASET_COLORS = {
    "simulation": COLORS["blue"],
    "allen": COLORS["orange"],
    "steinmetz": COLORS["slate"],
    "synphys": COLORS["purple"],
    "openscope": COLORS["teal"],
}
METHOD_COLORS = {
    "dynamics_aware": COLORS["teal"],
    "spectral": COLORS["blue"],
    "feedback hub": COLORS["slate"],
    "feedback_hub": COLORS["slate"],
    "high degree": COLORS["purple"],
    "high_degree": COLORS["purple"],
    "module bridge": COLORS["olive"],
    "module_bridge": COLORS["olive"],
    "cycle proxy": COLORS["rose"],
    "cycle_proxy": COLORS["rose"],
    "coverage greedy": COLORS["blue"],
    "coverage_greedy": COLORS["blue"],
    "random": COLORS["random"],
    "baseline": COLORS["baseline"],
}

MAIN_PALETTE = [COLORS[k] for k in ["teal", "blue", "purple", "orange", "olive", "rose", "slate"]]

HEATMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "ns_v53_diverging", ["#2166AC", "#F7F7F7", "#B2182B"], N=256
)
HEATMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "ns_v53_sequential", ["#F7FBFF", "#6BAED6", "#08519C"], N=256
)

# compatibility aliases used by older scripts
COLOR_COMPLEXITY_LOW = "#6B8FA8"
COLOR_COMPLEXITY_HIGH = COLORS["orange"]
COLOR_OPTIMIZED = COLORS["teal"]
COLOR_SPECTRAL = COLORS["blue"]
COLOR_HIGH_DEGREE = COLORS["purple"]
COLOR_RANDOM = COLORS["random"]
COLOR_BIOLOGY = COLORS["orange"]
COLOR_NEUTRAL = COLORS["mid_gray"]

@dataclass(frozen=True)
class FigureSpec:
    width_mm: float
    height_mm: float
    role: str
    @property
    def size_inches(self):
        return (self.width_mm / MM_PER_INCH, self.height_mm / MM_PER_INCH)

FIGURE_SPECS = {
    "compact": FigureSpec(54.0, PANEL_HEIGHT_MM, "compact leaf panel"),
    "small": FigureSpec(72.0, PANEL_HEIGHT_MM, "small leaf panel"),
    "single": FigureSpec(89.0, PANEL_HEIGHT_MM, "single-column leaf panel"),
    "onehalf": FigureSpec(136.0, PANEL_HEIGHT_MM, "wide leaf panel"),
    "double": FigureSpec(183.0, PANEL_HEIGHT_MM, "double-column strip panel"),
    "square": FigureSpec(72.0, PANEL_HEIGHT_MM, "square panel"),
    "matrix": FigureSpec(72.0, PANEL_HEIGHT_MM, "matrix / heatmap panel"),
    "forest": FigureSpec(89.0, PANEL_HEIGHT_MM, "forest / interval panel"),
    "s4a_method": FigureSpec(89.0, PANEL_HEIGHT_MM, "Stage4A method forest"),
    "s4a_gain": FigureSpec(89.0, PANEL_HEIGHT_MM, "Stage4A paired-gain forest"),
    "s4a_budget": FigureSpec(136.0, PANEL_HEIGHT_MM, "Stage4A state-budget curve"),
    "s4a_descriptor": FigureSpec(89.0, PANEL_HEIGHT_MM, "Stage4A descriptor forest"),
    "s4a_correlation": FigureSpec(89.0, PANEL_HEIGHT_MM, "Stage4A descriptor association"),
    "s4a_validation": FigureSpec(89.0, PANEL_HEIGHT_MM, "Stage4A validation scatter"),
    "s4a_composite_2x3": FigureSpec(183.0, 120.0, "compatibility composite only"),
    "s4a_composite_2x2": FigureSpec(136.0, 100.0, "compatibility composite only"),
    "os_paired": FigureSpec(54.0, PANEL_HEIGHT_MM, "paired subject comparison"),
    "os_mouse": FigureSpec(72.0, PANEL_HEIGHT_MM, "mouse-wise effect/value"),
    "os_group_ci": FigureSpec(54.0, PANEL_HEIGHT_MM, "group effect with CI"),
    "os_forest": FigureSpec(89.0, PANEL_HEIGHT_MM, "endpoint forest"),
    "os_line": FigureSpec(72.0, PANEL_HEIGHT_MM, "ordered line plot"),
    "os_repeat_mouse": FigureSpec(72.0, PANEL_HEIGHT_MM, "repeat distribution"),
    "os_multi_measure": FigureSpec(89.0, PANEL_HEIGHT_MM, "multi-measure plot"),
    "os_histogram": FigureSpec(72.0, PANEL_HEIGHT_MM, "null histogram"),
    "os_stacked_bar": FigureSpec(72.0, PANEL_HEIGHT_MM, "stacked subject bar"),
    "os_loo": FigureSpec(72.0, PANEL_HEIGHT_MM, "leave-one-out plot"),
}


def mm_to_inch(mm: float) -> float:
    return mm / MM_PER_INCH


def figure_size_mm(kind: str):
    spec = FIGURE_SPECS[kind]
    return spec.width_mm, spec.height_mm


def figure_size_inches(kind: str):
    return FIGURE_SPECS[kind].size_inches


def figure_size_table() -> pd.DataFrame:
    """Return the fixed figure-size contract used by legacy V4/V5 plotters."""
    return pd.DataFrame([
        {"kind": kind, "width_mm": spec.width_mm, "height_mm": spec.height_mm, "role": spec.role}
        for kind, spec in FIGURE_SPECS.items()
    ])


def choose_width_mm_from_ratio(ratio: float) -> float:
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = 1.35
    if ratio < 0.90:
        return 54.0
    if ratio < 1.25:
        return 72.0
    if ratio < 1.75:
        return 89.0
    if ratio < 2.45:
        return 136.0
    return 183.0


def set_paper_style(base_fontsize: float = FONT_SIZE_BASE) -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": base_fontsize,
        "axes.labelsize": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_TITLE_MAX,
        "axes.titleweight": "normal",
        "axes.labelcolor": COLORS["black"],
        "text.color": COLORS["black"],
        "axes.edgecolor": COLORS["black"],
        "axes.linewidth": LINEWIDTH_AXIS,
        "xtick.color": COLORS["black"],
        "ytick.color": COLORS["black"],
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "xtick.major.width": LINEWIDTH_AXIS,
        "ytick.major.width": LINEWIDTH_AXIS,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "legend.frameon": False,
        "lines.linewidth": LINEWIDTH_DATA,
        "lines.markersize": MARKER_SIZE,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
        "figure.max_open_warning": 100,
        "image.cmap": "viridis",
    })


def init_global_style(): set_paper_style()
def apply_style(): set_paper_style()
def use_style(): set_paper_style()
def set_style(): set_paper_style()


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_figure(kind: str = "single"):
    set_paper_style()
    spec = FIGURE_SPECS[kind]
    fig, ax = plt.subplots(figsize=spec.size_inches)
    fig._nature_width_mm = spec.width_mm
    fig._nature_height_mm = spec.height_mm
    fig._figure_kind = kind
    return fig, ax


def subplots(kind: str = "single", nrows=1, ncols=1, **kwargs):
    set_paper_style()
    spec = FIGURE_SPECS[kind]
    kwargs.pop("figsize", None)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=spec.size_inches, **kwargs)
    fig._nature_width_mm = spec.width_mm
    fig._nature_height_mm = spec.height_mm
    fig._figure_kind = kind
    return fig, axs


def create_standard_figure():
    return new_figure("single")


def style_axes(ax, frame: str = "open", xlabel: Optional[str] = None,
               ylabel: Optional[str] = None, title: Optional[str] = None,
               xgrid: bool = False, ygrid: bool = False,
               tick_direction: str = "out", **kwargs):
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH_AXIS)
        spine.set_color(COLORS["black"])
    if frame == "open":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
    elif frame == "box":
        for s in ax.spines.values():
            s.set_visible(True)
    elif frame == "none":
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        raise ValueError("frame must be 'open', 'box', or 'none'")
    ax.tick_params(direction=tick_direction, length=TICK_LENGTH, width=LINEWIDTH_AXIS,
                   colors=COLORS["black"], labelsize=FONT_SIZE_TICK)
    ax.grid(False)
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    if title is not None: ax.set_title(title, fontsize=FONT_SIZE_TITLE_MAX)
    return ax


def apply_standard_axis_settings(ax, xlabel="", ylabel="", label_fontsize=None, tick_labelsize=None):
    return style_axes(ax, frame="open", xlabel=xlabel or None, ylabel=ylabel or None)


def remove_title(ax): ax.set_title("")


def add_zero_line(ax, orientation: str = "h", color: str = COLORS["mid_gray"],
                  ls: str = "--", lw: float = LINEWIDTH_REFERENCE):
    if orientation == "h": return ax.axhline(0, color=color, ls=ls, lw=lw, zorder=0)
    if orientation == "v": return ax.axvline(0, color=color, ls=ls, lw=lw, zorder=0)
    raise ValueError("orientation must be 'h' or 'v'")


def add_panel_label(ax, label: str, x=-0.16, y=1.04):
    return ax.text(x, y, label, transform=ax.transAxes, fontweight="bold",
                   fontsize=FONT_SIZE_PANEL, va="top", ha="left",
                   color=COLORS["black"], clip_on=False)


def legend_outside(ax, where: str = "right", ncol: int = 1, title=None, **kwargs):
    if where == "right":
        return ax.legend(loc="upper left", bbox_to_anchor=(1.02,1.0), borderaxespad=0,
                         ncol=ncol, title=title, frameon=False, **kwargs)
    if where == "top":
        return ax.legend(loc="lower center", bbox_to_anchor=(0.5,1.02), borderaxespad=0,
                         ncol=ncol, title=title, frameon=False, **kwargs)
    raise ValueError("where must be 'right' or 'top'")


def legend_shared(fig, handles, labels, where="top", ncol=3, **kwargs):
    loc, anchor = ("upper center", (0.5,1.01)) if where == "top" else ("lower center", (0.5,-0.01))
    return fig.legend(handles, labels, loc=loc, bbox_to_anchor=anchor, ncol=ncol, frameon=False, **kwargs)


def plot_mean_ci(ax, x, mean, lower, upper, color=COLORS["blue"], label=None,
                 lw=LINEWIDTH_MAIN, alpha=ALPHA_BAND):
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0)
    return ax.plot(x, mean, color=color, lw=lw, marker='o', mfc=color, mec=color, label=label)


def _looks_like_old_panel_letter(text: Text, ax) -> bool:
    s = (text.get_text() or "").strip()
    if not re.fullmatch(r"[a-z]", s): return False
    try:
        x, y = text.get_position()
        if text.get_transform() == ax.transAxes and x < 0.18 and y > 0.82: return True
    except Exception: pass
    return False

_SUBSCRIPT_MAP = {
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9",
    "₊":"+","₋":"-","₌":"=","₍":"(","₎":")","ₐ":"a","ₑ":"e","ₒ":"o","ₓ":"x","ₕ":"h",
    "ₖ":"k","ₗ":"l","ₘ":"m","ₙ":"n","ₚ":"p","ₛ":"s","ₜ":"t","ᵢ":"i","ⱼ":"j","ᵣ":"r","ᵤ":"u","ᵥ":"v",
}
_SUBSCRIPT_CHARS = "".join(map(re.escape, _SUBSCRIPT_MAP.keys()))

def _sanitize_math_text(s: str) -> str:
    if not s: return s
    s = s.replace("∪", r"$\cup$")
    def repl(m):
        raw = m.group(0)
        return "$_{" + "".join(_SUBSCRIPT_MAP.get(ch, ch) for ch in raw) + "}$"
    if "$" not in s:
        s = re.sub(f"[{_SUBSCRIPT_CHARS}]+", repl, s)
    return s


def _normalize_single_axis_geometry(fig: Figure):
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if len(axes) == 1:
        ax = axes[0]
        rots=[]; labels=[]
        for t in ax.get_xticklabels():
            try: rots.append(abs(float(t.get_rotation())))
            except Exception: pass
            labels.append(t.get_text() or "")
        long_x = any(r > 5 for r in rots) or max([len(x) for x in labels]+[0]) > 14
        bottom = 0.30 if long_x else 0.20
        ax.set_position([0.18, bottom, 0.77, 0.94-bottom])
    elif len(axes) == 2:
        poss=[ax.get_position() for ax in axes]; widths=[p.width for p in poss]
        if min(widths) < 0.10:
            main_i=int(np.argmax(widths)); cb_i=1-main_i
            axes[main_i].set_position([0.16,0.20,0.69,0.74])
            axes[cb_i].set_position([0.88,0.20,0.025,0.74])


def enforce_nature_figure(fig: Figure, *, strip_titles: Optional[bool]=None,
                          strip_panel_letters: Optional[bool]=None,
                          normalize_geometry: Optional[bool]=None) -> Figure:
    if strip_titles is None: strip_titles = os.environ.get("NATURE_STRIP_TITLES","1") != "0"
    if strip_panel_letters is None: strip_panel_letters = os.environ.get("NATURE_STRIP_PANEL_LETTERS","1") != "0"
    if normalize_geometry is None: normalize_geometry = os.environ.get("NATURE_NORMALIZE_AXES","0") != "0"
    fig.set_facecolor("white")
    if getattr(fig,"_suptitle",None) is not None and strip_titles:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        if strip_titles: ax.set_title("")
        try: ax.grid(False)
        except Exception: pass
        is_matrix = bool(getattr(ax,"images",[]))
        style_axes(ax, frame="box" if is_matrix else "open")
        for lab in [ax.xaxis.label, ax.yaxis.label]:
            try:
                lab.set_text(_sanitize_math_text(lab.get_text()))
                lab.set_fontsize(FONT_SIZE_LABEL); lab.set_fontfamily(FONT_FAMILY)
            except Exception: pass
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            try:
                t.set_text(_sanitize_math_text(t.get_text()))
                t.set_fontsize(FONT_SIZE_TICK); t.set_fontfamily(FONT_FAMILY)
            except Exception: pass
        leg=ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                try:
                    t.set_text(_sanitize_math_text(t.get_text()))
                    t.set_fontsize(FONT_SIZE_LEGEND); t.set_fontfamily(FONT_FAMILY)
                except Exception: pass
            try: leg.set_frame_on(False)
            except Exception: pass
        for txt in ax.texts:
            if strip_panel_letters and _looks_like_old_panel_letter(txt,ax):
                txt.set_visible(False); continue
            try:
                txt.set_text(_sanitize_math_text(txt.get_text()))
                txt.set_fontsize(min(float(txt.get_fontsize()), FONT_SIZE_ANNOT))
                txt.set_fontfamily(FONT_FAMILY)
            except Exception: pass
    for txt in fig.texts:
        try:
            txt.set_text(_sanitize_math_text(txt.get_text()))
            txt.set_fontsize(min(float(txt.get_fontsize()), FONT_SIZE_ANNOT))
            txt.set_fontfamily(FONT_FAMILY)
        except Exception: pass
    if normalize_geometry: _normalize_single_axis_geometry(fig)
    return fig


def strip_all_titles(fig):
    for ax in fig.axes: ax.set_title("")
    if getattr(fig,"_suptitle",None) is not None: fig._suptitle.set_text("")


def align_ylabels(fig):
    try: fig.align_ylabels()
    except Exception: pass


def finalize_figure(fig: Figure, strip_titles: bool=True):
    if strip_titles: strip_all_titles(fig)
    align_ylabels(fig)
    return enforce_nature_figure(fig, strip_titles=strip_titles)


def _write_size_sidecar(fig, outstem: Path, dpi: int):
    w,h=fig.get_size_inches()
    kind=getattr(fig,"_figure_kind",None)
    side={
        "figure_kind":kind,
        "width_mm":round(w*MM_PER_INCH,4), "height_mm":round(h*MM_PER_INCH,4),
        "dpi":dpi, "body_font_pt":FONT_SIZE_BASE, "label_font_pt":FONT_SIZE_LABEL,
        "tick_font_pt":FONT_SIZE_TICK, "legend_font_pt":FONT_SIZE_LEGEND,
        "panel_letter_pt":FONT_SIZE_PANEL, "bbox_inches":None,
        "leaf_panel_letters_stripped":True,
    }
    outstem.with_suffix(".size.json").write_text(json.dumps(side,indent=2),encoding="utf-8")


def save_figure(fig: Figure, outstem, dpi: int=OUTPUT_DPI, save_png: bool=True,
                save_pdf: bool=True, transparent: bool=False, close: bool=False,
                **kwargs):
    outstem=Path(outstem); ensure_dir(outstem.parent)
    enforce_nature_figure(fig)
    outputs=[]
    common=dict(bbox_inches=None, pad_inches=0, transparent=transparent, facecolor="white")
    if save_png:
        p=outstem.with_suffix(".png"); fig.savefig(p,dpi=dpi,**common); outputs.append(p)
    if save_pdf:
        p=outstem.with_suffix(".pdf"); fig.savefig(p,**common); outputs.append(p)
    _write_size_sidecar(fig,outstem,dpi)
    if close: plt.close(fig)
    return outputs


def save_plot_data(df: pd.DataFrame, outcsv, index: bool=False):
    p=Path(outcsv); ensure_dir(p.parent); df.to_csv(p,index=index,encoding="utf-8-sig"); return p


def export_panel_bundle(fig: Figure, outstem, data=None, dpi: int=OUTPUT_DPI,
                        save_png: bool=True, save_pdf: bool=True,
                        transparent: bool=False, close: bool=False, **kwargs):
    outstem=Path(outstem)
    saved=save_figure(fig,outstem,dpi=dpi,save_png=save_png,save_pdf=save_pdf,
                      transparent=transparent,close=False)
    result={p.suffix.lstrip('.'):p for p in saved}; result['size']=outstem.with_suffix('.size.json')
    if data is not None:
        try: result['csv']=save_plot_data(data,outstem.with_suffix('.csv'))
        except Exception: pass
    if close: plt.close(fig)
    return result

set_paper_style()
