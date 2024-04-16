# -*- coding: utf-8 -*-

# Valores por defecto de estilo:
SCATTER_MARKER = "o"
SCATTER_MARKERSIZE = 3
SCATTER_MARKEREDGEWIDTH = 0
SCATTER_ALPHA = 0.4
LINE_LS = "-"
LINE_LINEWIDTH = 0.5
LINE_MARKER = "o"
LINE_MARKERSIZE = 2.0
LINE_MARKEREDGEWIDTH = 0.0
LINE_ALPHA = 1.0
LINE_XSCALE = "log"
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 12

DEFAULT_PLOT_STYLE = {}

DEFAULT_PLOT_STYLE["sample"] = {
    "line_kwargs": {
        "linewidth": 0,
        "ls": "none",
        "color": "#000000",
        "marker": SCATTER_MARKER,
        "markersize": SCATTER_MARKERSIZE,
        "markeredgewidth": SCATTER_MARKEREDGEWIDTH,
        "markeredgecolor": "#000000",
        "markerfacecolor": "#000000",
        "alpha": SCATTER_ALPHA,
        "label": "Virtual sources"},
    "title": None,
    "title_fontdict": {
        "size": TITLE_FONTSIZE,
        "color": "#000000"},
    "xlabel": None,
    "xlabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "ylabel": None,
    "ylabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "xscale": "linear",
    "yscale": "linear"}
DEFAULT_PLOT_STYLE["residue"] = {
    "line_kwargs": {
        "linewidth": 0,
        "ls": "none",
        "color": "#000000",
        "marker": SCATTER_MARKER,
        "markersize": SCATTER_MARKERSIZE,
        "markeredgewidth": SCATTER_MARKEREDGEWIDTH,
        "markeredgecolor": "#000000",
        "markerfacecolor": "#000000",
        "alpha": SCATTER_ALPHA,
        "label": "Virtual sources"},
    "title": None,
    "title_fontdict": {
        "size": TITLE_FONTSIZE,
        "color": "#000000"},
    "xlabel": None,
    "xlabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "ylabel": None,
    "ylabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "xscale": "linear",
    "yscale": "linear"}

DEFAULT_PLOT_STYLE["fitness"] = {
    "line_kwargs": {
        "linewidth": LINE_LINEWIDTH,
        "ls": LINE_LS,
        "color": "#000000",
        "marker": LINE_MARKER,
        "markersize": LINE_MARKERSIZE,
        "markeredgewidth": LINE_MARKEREDGEWIDTH,
        "markeredgecolor": "#000000",
        "markerfacecolor": "#000000",
        "alpha": LINE_ALPHA,
        "label": "Fitness"},
    "title": None,
    "title_fontdict": {
        "size": TITLE_FONTSIZE,
        "color": "#000000"},
    "xlabel": "Epoch",
    "xlabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "ylabel": "Fitness",
    "ylabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "xscale": LINE_XSCALE,
    "yscale": "log"}
DEFAULT_PLOT_STYLE["alpha"] = {
    "line_kwargs": {
        "linewidth": LINE_LINEWIDTH,
        "ls": LINE_LS,
        "color": "#FF0000",
        "marker": LINE_MARKER,
        "markersize": LINE_MARKERSIZE,
        "markeredgewidth": LINE_MARKEREDGEWIDTH,
        "markeredgecolor": "#000000",
        "markerfacecolor": "#FF0000",
        "alpha": LINE_ALPHA,
        "label": "Alpha"},
    "title": None,
    "title_fontdict": {
        "size": TITLE_FONTSIZE,
        "color": "#000000"},
    "xlabel": "Epoch",
    "xlabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "ylabel": "Alpha",
    "ylabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#FF0000"},
    "xscale": LINE_XSCALE,
    "yscale": "linear"}
DEFAULT_PLOT_STYLE["beta"] = {
    "line_kwargs": {
        "linewidth": LINE_LINEWIDTH,
        "ls": LINE_LS,
        "color": "#0000FF",
        "marker": LINE_MARKER,
        "markersize": LINE_MARKERSIZE,
        "markeredgewidth": LINE_MARKEREDGEWIDTH,
        "markeredgecolor": "#000000",
        "markerfacecolor": "#0000FF",
        "alpha": LINE_ALPHA,
        "label": "Beta"},
    "title": None,
    "title_fontdict": {
        "size": TITLE_FONTSIZE,
        "color": "#000000"},
    "xlabel": "Epoch",
    "xlabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "ylabel": "Beta",
    "ylabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#0000FF"},
    "xscale": LINE_XSCALE,
    "yscale": "linear"}
DEFAULT_PLOT_STYLE["mean_displacement"] = {
    "line_kwargs": {
        "linewidth": LINE_LINEWIDTH,
        "ls": LINE_LS,
        "color": "#000000",
        "marker": LINE_MARKER,
        "markersize": LINE_MARKERSIZE,
        "markeredgewidth": LINE_MARKEREDGEWIDTH,
        "markeredgecolor": "#000000",
        "markerfacecolor": "#000000",
        "alpha": LINE_ALPHA,
        "label": "Mean displacement"},
    "title": None,
    "title_fontdict": {
        "size": TITLE_FONTSIZE,
        "color": "#000000"},
    "xlabel": "Epoch",
    "xlabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "ylabel": "Mean displacement",
    "ylabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "xscale": LINE_XSCALE,
    "yscale": "log"}
DEFAULT_PLOT_STYLE["max_displacement"] = {
    "line_kwargs": {
        "linewidth": LINE_LINEWIDTH,
        "ls": LINE_LS,
        "color": "#AAAAAA",
        "marker": LINE_MARKER,
        "markersize": LINE_MARKERSIZE,
        "markeredgewidth": LINE_MARKEREDGEWIDTH,
        "markeredgecolor": "#000000",
        "markerfacecolor": "#AAAAAA",
        "alpha": LINE_ALPHA,
        "label": "Max. displacement"},
    "title": None,
    "title_fontdict": {
        "size": TITLE_FONTSIZE,
        "color": "#000000"},
    "xlabel": "Epoch",
    "xlabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "ylabel": "Max. displacement",
    "ylabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#AAAAAA"},
    "xscale": LINE_XSCALE,
    "yscale": "log"}
DEFAULT_PLOT_STYLE["success_rate"] = {
    "line_kwargs": {
        "linewidth": LINE_LINEWIDTH,
        "ls": LINE_LS,
        "color": "#005599",
        "marker": LINE_MARKER,
        "markersize": LINE_MARKERSIZE,
        "markeredgewidth": LINE_MARKEREDGEWIDTH,
        "markeredgecolor": "#000000",
        "markerfacecolor": "#005599",
        "alpha": LINE_ALPHA,
        "label": "Success rate"},
    "title": None,
    "title_fontdict": {
        "size": TITLE_FONTSIZE,
        "color": "#000000"},
    "xlabel": "Epoch",
    "xlabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "ylabel": "Success rate",
    "ylabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#005599"},
    "xscale": LINE_XSCALE,
    "yscale": "linear"}
DEFAULT_PLOT_STYLE["global_scale"] = {
    "line_kwargs": {
        "linewidth": LINE_LINEWIDTH,
        "ls": LINE_LS,
        "color": "#009955",
        "marker": "",
        "markersize": LINE_MARKERSIZE,
        "markeredgewidth": LINE_MARKEREDGEWIDTH,
        "markeredgecolor": "#000000",
        "markerfacecolor": "#009955",
        "alpha": LINE_ALPHA,
        "label": "Global scale"},
    "title": None,
    "title_fontdict": {
        "size": TITLE_FONTSIZE,
        "color": "#000000"},
    "xlabel": "Epoch",
    "xlabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#000000"},
    "ylabel": "Global scale",
    "ylabel_fontdict": {
        "size": LABEL_FONTSIZE,
        "color": "#009955"},
    "xscale": LINE_XSCALE,
    "yscale": "log"}