"""MIOS V6 — Dashboard 2's terminal chart: NIFTY ‖ ATM Call ‖ ATM Put.

    ┌──────────────────────────┬──────────────────┐
    │                          │    ATM CALL      │
    │        NIFTY  (60%)      ├──────────────────┤
    │                          │    ATM PUT       │
    └──────────────────────────┴──────────────────┘

**One figure, not three.** This is the decision that makes the layout work.
Streamlit columns would give three independent Plotly figures, and Plotly can
only synchronise axes *within* a figure — so three figures means three
independently-scrolling charts, which is precisely what the terminal must not
be. A single figure with a `rowspan` cell and `matches="x"` on every axis gives
the requested proportions **and** real synchronised zoom, pan and crosshair.

`matches="x"` rather than `shared_xaxes=True`: the latter only links subplots
within a column, and NIFTY is in a different column from the option legs.

Levels are drawn on the NIFTY panel only. Overlaying a spot-derived stop on an
option's price series would put a line at a price that series can never trade —
authoritative-looking and meaningless.

Tinting is applied as a below-layer rectangle rather than a plot background so
the candles keep their own colours. A chart whose body colour competes with its
candles is harder to read, not easier.

Not a pure module — it builds a Plotly figure. It computes no market logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

_UP, _DOWN = "#26a69a", "#ef5350"

#: level → (colour, dash, width). Drawn on the NIFTY panel only.
LEVELS = {
    "entry":       ("#00ff88", "solid",   1.6),
    "stop":        ("#ff4444", "dash",    1.5),
    "trail":       ("#a78bfa", "dot",     1.5),
    "target":      ("#7fe8b0", "dashdot", 1.3),
    "support":     ("#17c98b", "dot",     1.2),
    "resistance":  ("#ff8c8c", "dot",     1.2),
    "war_zone":    ("#ffcc33", "dash",    1.4),
    "liquidity":   ("#4da6ff", "dot",     1.1),
    "vwap":        ("#c9b6ec", "solid",   1.2),
    "poc":         ("#ffe066", "dash",    1.3),
    "vah":         ("#7dffb0", "dot",     1.1),
    "val":         ("#ff8c8c", "dot",     1.1),
}

LEVEL_LABEL = {
    "entry": "Entry", "stop": "Stop", "trail": "Trail", "target": "Target",
    "support": "Support", "resistance": "Resistance", "war_zone": "War Zone",
    "liquidity": "Liquidity", "vwap": "VWAP", "poc": "POC", "vah": "VAH",
    "val": "VAL",
}

#: higher-timeframe POCs get their own dimmer treatment so they never compete
#: with today's actionable levels
HTF_COLOUR = "#9fb0c4"

#: decision state → the marker drawn on the NIFTY panel at the current bar
SIGNAL_MARKER = {
    "ENTER": ("▲ ENTER", "#00ff88", "triangle-up"),
    "ENTRY_READY": ("◆ READY", "#ffcc33", "diamond"),
    "FLOOR_CONFIRMED": ("▲ FLOOR", "#00ff88", "triangle-up"),
    "CEILING_CONFIRMED": ("▼ CEILING", "#ff4444", "triangle-down"),
    "EXIT": ("■ EXIT", "#ff9500", "square"),
    "ABORT": ("✕ ABORT", "#ff2d55", "x"),
    "TRAIL": ("↗ TRAIL", "#a78bfa", "arrow-up"),
    "SCALE_IN": ("＋ SCALE IN", "#00ff88", "cross"),
    "SCALE_OUT": ("－ SCALE OUT", "#ffcc33", "cross"),
}


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _ohlc(df) -> Optional[Dict[str, Any]]:
    """OHLC + a time axis, whatever the frame calls its columns. `None` when
    the frame cannot be charted, so the caller reports it instead of drawing an
    empty axis."""
    if df is None or getattr(df, "empty", True):
        return None
    cols = {str(c).lower(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return df[cols[n]]
        return None

    close = col("close", "ltp", "price")
    if close is None:
        return None

    # ── datetime BEFORE timestamp. The NIFTY frame carries both: `timestamp`
    # is raw epoch seconds, `datetime` is the tz-aware IST column. Preferring
    # `timestamp` plotted NIFTY against integers while the option legs — which
    # only have `datetime` — plotted against datetimes, so `matches="x"` linked
    # two incompatible axis types and the shared timeline silently broke. The
    # shared axis is the entire point of this chart.
    x = col("datetime", "time", "date", "timestamp")
    if x is None:
        x = df.index
    x = _as_time(x)

    o, h, l = col("open"), col("high"), col("low")
    return {"x": x, "open": o, "high": h, "low": l, "close": close,
            "volume": col("volume", "vol"),
            "line_only": o is None or h is None or l is None}


def _as_time(x):
    """Whatever the frame carries → tz-NAIVE IST datetimes.

    Two separate problems, one function.

    **Epoch seconds.** A frame that only carries `timestamp` would draw on a
    numeric axis that cannot line up with the datetime axis on the panel beside
    it, so `matches="x"` silently fails to link them.

    **The timezone.** Plotly.js has no timezone support: plotly.py serialises a
    tz-aware timestamp by converting it to UTC and dropping the offset, so an
    IST series renders 5½ hours early. That is why the axis read 04:00–11:00
    while the market was at 09:48 — the data was right and the label was UTC.
    Converting to IST and *then* dropping the tz makes the wall-clock number
    itself IST, which is the only thing Plotly will render faithfully.
    """
    try:
        import pandas as pd

        kind = getattr(getattr(x, "dtype", None), "kind", "")
        if kind in ("i", "u", "f"):
            first = float(x.iloc[0]) if hasattr(x, "iloc") else float(list(x)[0])
            # plausible epoch seconds (1970-2100), not a row index
            if not 1e8 < first < 4e9:
                return x
            x = pd.to_datetime(x, unit="s", utc=True)
        elif kind != "M" and not hasattr(x, "dt"):
            x = pd.to_datetime(pd.Series(list(x)), errors="coerce")

        s = x if hasattr(x, "dt") else pd.Series(x)
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("Asia/Kolkata")
        else:
            # naive values off the wire are already IST — say so before
            # converting, or the conversion shifts them a second time
            s = s.dt.tz_localize("Asia/Kolkata", ambiguous="NaT",
                                 nonexistent="NaT")
        return s.dt.tz_localize(None)
    except Exception:
        return x


def terminal_chart(nifty_df=None, call_df=None, put_df=None,
                   levels: Optional[Dict[str, Any]] = None,
                   htf_levels: Optional[Dict[str, Any]] = None,
                   call_label: str = "ATM Call", put_label: str = "ATM Put",
                   tint: Optional[str] = None,
                   dominance: str = "neutral",
                   signal: Optional[Dict[str, Any]] = None,
                   height: int = 660,
                   window_minutes: Optional[int] = None,
                   call_levels: Optional[Dict[str, Any]] = None,
                   put_levels: Optional[Dict[str, Any]] = None,
                   call_zones: Optional[Sequence[Dict[str, Any]]] = None,
                   put_zones: Optional[Sequence[Dict[str, Any]]] = None,
                   overlays: Optional[Sequence[Dict[str, Any]]] = None):
    """The three-panel terminal. Returns `(figure, notes)`; `notes` names any
    series that could not be drawn."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── today's session only. The API is asked for a `days_back` WINDOW, not
    # a session count, so every frame arrives carrying part of the previous
    # day — which squeezes the live session into a sliver at the right edge and
    # makes the previous close look like an intraday level.
    from ..clock import today_slice
    panels = [("NIFTY", today_slice(nifty_df)),
              (call_label, today_slice(call_df)),
              (put_label, today_slice(put_df))]
    parsed = [(name, _ohlc(df)) for name, df in panels]
    notes = [name for name, p in parsed if p is None]

    # ── one clock for all three panels. Every chart is reindexed onto the
    # union of their timestamps, so candle *n* is the same minute everywhere
    # and a minute a leg did not trade shows as a gap rather than sliding
    # every later candle one slot left.
    timeline = master_timeline(parsed)
    parsed = [(name, align(p, timeline)) for name, p in parsed]

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        column_widths=[0.60, 0.40], row_heights=[0.5, 0.5],
        horizontal_spacing=0.045, vertical_spacing=0.05,
        subplot_titles=[n for n, _ in parsed])

    positions = [(1, 1), (1, 2), (2, 2)]
    for (name, p), (r, c) in zip(parsed, positions):
        if p is None:
            continue
        _add_series(go, fig, name, p, r, c)

    # ── overlays: each carries its OWN style and the engine that produced it,
    # so a level on screen is always traceable to the thing that claimed it.
    # Drawn on the NIFTY panel only — a spot-derived level on a premium series
    # marks a price that series can never trade.
    for o in (overlays or []):
        v = _f(o.get("price"))
        if v is None:
            continue
        fig.add_hline(y=v, row=1, col=1,
                      line_width=o.get("width", 1.0),
                      line_dash=o.get("dash", "dot"),
                      line_color=o.get("colour", "#9fb0c4"),
                      annotation_text=f"{o.get('label', '')} {v:,.0f}",
                      annotation_position="right",
                      annotation_font=dict(size=9,
                                           color=o.get("colour", "#9fb0c4")))

    # ── levels: NIFTY panel only (legacy styled-key path) ──
    for key, price in (levels or {}).items():
        v = _f(price)
        if v is None or key not in LEVELS:
            continue
        colour, dash, width = LEVELS[key]
        fig.add_hline(y=v, row=1, col=1, line_width=width, line_dash=dash,
                      line_color=colour,
                      annotation_text=f"{LEVEL_LABEL[key]} {v:,.0f}",
                      annotation_position="right",
                      annotation_font=dict(size=9, color=colour))

    # higher-timeframe POCs — dimmer, so they never compete with today's
    for label, price in sorted((htf_levels or {}).items(),
                               key=lambda kv: -(_f(kv[1]) or 0))[:8]:
        v = _f(price)
        if v is None:
            continue
        fig.add_hline(y=v, row=1, col=1, line_width=1, line_dash="dot",
                      line_color=HTF_COLOUR,
                      annotation_text=f"{label} {v:,.0f}",
                      annotation_position="left",
                      annotation_font=dict(size=8, color=HTF_COLOUR))

    # ── the option panels get their OWN levels and VOB zones, in premium.
    # A spot-derived stop drawn on a premium series marks a price that series
    # can never trade — authoritative-looking and meaningless. These come from
    # the legs' own engines, so the number belongs to the axis it sits on.
    _leg_overlay(fig, call_levels, call_zones, 1, 2)
    _leg_overlay(fig, put_levels, put_zones, 2, 2)

    _add_signal(go, fig, parsed[0][1], signal)
    _tint(fig, tint, dominance)

    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=26, b=8),
        paper_bgcolor="#0b0f16", plot_bgcolor="#0b0f16",
        font=dict(color="#edf3f9", size=10), showlegend=False,
        hovermode="x unified", dragmode="pan",
        # ── the view survives the rerun. Streamlit rebuilds the whole figure
        # every cycle, which resets zoom, pan and crosshair to default — so a
        # trader who zoomed into 10:15–11:20 was thrown back out to the full
        # session a second later. A constant `uirevision` tells Plotly to keep
        # the user's view across redraws. It is keyed on the zoom level so the
        # ➕/➖ buttons still take effect: only a deliberate change moves it.
        uirevision=f"terminal:{window_minutes}")
    # ── the shared crosshair readout. `hovermode="x unified"` alone gathers
    # traces within ONE subplot; `hoversubplots="axis"` extends the same hover
    # across every panel on the linked axis, so hovering 10:48 reads 10:48 on
    # NIFTY, CALL and PUT at once. Older Plotly ignores the key rather than
    # raising, so the chart degrades to per-panel hover instead of failing.
    try:
        fig.update_layout(hoversubplots="axis")
    except Exception:
        pass
    # ── the synchronisation. `matches` links every panel to NIFTY's axis, so
    # zoom / pan / scroll on any of them moves all three together. Independent
    # scrolling is the one thing this layout must not allow.
    fig.update_xaxes(matches="x", rangeslider_visible=False,
                     gridcolor="#161b22", showspikes=True,
                     spikecolor="#3a4757", spikethickness=1,
                     spikemode="across", spikesnap="cursor")
    fig.update_yaxes(gridcolor="#161b22", side="right", showspikes=False)

    # ── the zoom window, set by the buttons. Applied to the master axis only:
    # every other panel carries `matches="x"`, so they follow. The y-axes are
    # left to autorange so a zoomed-in window rescales to the prices actually
    # in it rather than staying flat inside the whole session's range.
    rng = x_range(_last_x(parsed), window_minutes)
    if rng:
        fig.update_xaxes(range=rng, row=1, col=1)

    # ── each panel's price axis, fitted to the bars on screen. Plotly's
    # autorange fits the whole trace rather than the visible window, so without
    # this a zoomed-in view kept the full session's height and the candles
    # stayed flat — the zoom moved time and nothing else.
    for (_name, p), (r, c) in zip(parsed, positions):
        if p is None:
            continue
        lo_hi = price_range(p.get("low") if p.get("low") is not None
                            else p["close"],
                            p.get("high") if p.get("high") is not None
                            else p["close"],
                            p["x"], rng)
        if lo_hi:
            fig.update_yaxes(range=lo_hi, row=r, col=c)

    for a in fig.layout.annotations:
        if a.text in [n for n, _ in parsed]:
            a.font.size = 11.5
            a.font.color = "#cfd9e6"
            a.xanchor = "left"
    return fig, notes


#: VOB zone status → (fill, edge). Building zones read loudest; a faded zone
#: is drawn but must not compete with one that is still being defended.
ZONE_TONE = {
    "BUILDING": ("rgba(38,166,154,.16)", "#26a69a"),
    "INTACT":   ("rgba(77,166,255,.11)", "#4da6ff"),
    "FADING":   ("rgba(159,176,196,.08)", "#7d8b9c"),
    "BREAKING": ("rgba(239,83,80,.14)", "#ef5350"),
}


def _leg_overlay(fig, levels, zones, row: int, col: int) -> None:
    """One option panel's own levels and VOB zones, in premium terms."""
    for key, price in (levels or {}).items():
        v = _f(price)
        if v is None or key not in LEVELS:
            continue
        colour, dash, width = LEVELS[key]
        fig.add_hline(y=v, row=row, col=col, line_width=width, line_dash=dash,
                      line_color=colour,
                      annotation_text=f"{LEVEL_LABEL[key]} ₹{v:,.2f}",
                      annotation_position="left",
                      annotation_font=dict(size=8, color=colour))

    for z in list(zones or [])[:6]:
        lo, hi = _f(z.get("lower")), _f(z.get("upper"))
        if lo is None or hi is None or hi <= lo:
            continue
        fill, edge = ZONE_TONE.get(_s(z.get("status")),
                                   ZONE_TONE["INTACT"])
        fig.add_hrect(y0=lo, y1=hi, row=row, col=col, layer="below",
                      fillcolor=fill, line_width=1, line_color=edge,
                      opacity=1.0)


def _s(v) -> str:
    return str(v or "").strip().upper()


def _last_x(parsed) -> Any:
    """The newest bar across every panel that drew.

    Not NIFTY's alone — on a day the index frame is missing, the option legs
    still define a timeline, and the zoom buttons should still work.
    """
    latest = None
    for _name, p in (parsed or []):
        if not p:
            continue
        try:
            v = list(p["x"])[-1]
        except Exception:
            continue
        if v is None:
            continue
        try:
            if latest is None or v > latest:
                latest = v
        except TypeError:            # mixed axis types — take the first we saw
            continue
    return latest


def _add_series(go, fig, name: str, p: Dict[str, Any], row: int, col: int):
    """Candles (or a line when OHLC is unavailable), with volume and the
    hover payload the crosshair reads."""
    hover = ("<b>%{x|%H:%M}</b><br>"
             + f"{name} %{{y:,.2f}}"
             + ("<br>Vol %{customdata:,.0f}" if p.get("volume") is not None
                else "") + "<extra></extra>")
    custom = p["volume"] if p.get("volume") is not None else None

    if p["line_only"]:
        fig.add_trace(go.Scatter(x=p["x"], y=p["close"], name=name,
                                 mode="lines", customdata=custom,
                                 hovertemplate=hover,
                                 line=dict(color="#7fb4ff", width=1.4)),
                      row=row, col=col)
        return

    fig.add_trace(go.Candlestick(
        x=p["x"], open=p["open"], high=p["high"], low=p["low"],
        close=p["close"], name=name, showlegend=False,
        customdata=custom,
        increasing_line_color=_UP, decreasing_line_color=_DOWN,
        increasing_fillcolor=_UP, decreasing_fillcolor=_DOWN),
        row=row, col=col)

    if p.get("volume") is not None:
        # volume as a faint overlay on its own scale — present for the
        # crosshair readout without stealing the price axis
        bars = volume_bars(p["volume"], p["low"], p["high"])
        if bars is not None:
            base, heights = bars
            fig.add_trace(go.Bar(x=p["x"], y=heights, base=base, name="vol",
                                 marker_color="rgba(120,140,170,.22)",
                                 hoverinfo="skip", showlegend=False),
                          row=row, col=col)


#: minutes of session visible at each zoom step; `None` is the whole session.
#: Ordered widest-last so "contract" walks right and "expand" walks left.
ZOOM_STEPS = (15, 30, 60, 120, 240, None)


def zoom_step(current: Optional[int], direction: int) -> Optional[int]:
    """The next zoom level in `direction` (+1 expands, -1 contracts).

    Clamped at both ends rather than wrapping — a button that silently jumps
    from the tightest view to the whole session looks like a misclick.
    """
    try:
        i = ZOOM_STEPS.index(current)
    except ValueError:
        i = len(ZOOM_STEPS) - 1          # unknown value → the full session
    i = max(0, min(len(ZOOM_STEPS) - 1, i - int(direction)))
    return ZOOM_STEPS[i]


def zoom_label(current: Optional[int]) -> str:
    if current is None:
        return "Full session"
    if current % 60 == 0:
        return f"Last {current // 60}h"
    return f"Last {current}m"


def x_range(last_x, minutes: Optional[int]):
    """`[start, end]` for the shared time axis, anchored at the newest bar.

    Anchored at the right edge, not the left: zooming in on a live chart should
    keep the price that is trading now on screen. `None` when the window cannot
    be computed, so the caller leaves the axis to autorange rather than pinning
    it to a bogus range.
    """
    if minutes is None or last_x is None:
        return None
    try:
        from datetime import datetime, timedelta
        end = last_x
        if not isinstance(end, datetime):
            try:
                import pandas as pd
                end = pd.Timestamp(end).to_pydatetime()
            except Exception:
                return None
        return [end - timedelta(minutes=int(minutes)), end]
    except Exception:
        return None


def master_timeline(parsed) -> List[Any]:
    """The union of every panel's timestamps, sorted — one clock for all three.

    Without this each panel plots its own x series, and three series of
    different lengths put candle *n* at a different minute on each panel. The
    axes would still be linked, so nothing would look broken; the CALL panel
    would simply be showing 10:47 while NIFTY showed 10:48. That is the failure
    mode a trader cannot see and cannot recover from.
    """
    stamps = set()
    for _name, p in (parsed or []):
        if not p:
            continue
        try:
            stamps.update(x for x in list(p["x"]) if x is not None and x == x)
        except Exception:
            continue
    try:
        return sorted(stamps)
    except TypeError:                      # mixed axis types — refuse to guess
        return []


def align(p: Optional[Dict[str, Any]], timeline: Sequence[Any]):
    """One panel's OHLCV reindexed onto the master timeline.

    A minute the leg did not trade becomes NaN, which Plotly renders as a gap.
    The alternative — letting the series keep its own shorter index — slides
    every later candle one slot left and silently misaligns the panels.
    """
    if p is None or not timeline:
        return p
    try:
        import pandas as pd

        idx = pd.Index(list(timeline))
        out = dict(p)
        base = pd.Index(list(p["x"]))
        for key in ("open", "high", "low", "close", "volume"):
            series = p.get(key)
            if series is None:
                continue
            s = pd.Series(list(series), index=base)
            s = s[~s.index.duplicated(keep="last")]
            out[key] = s.reindex(idx)
        out["x"] = idx
        return out
    except Exception:
        return p


#: breathing room above and below the visible extremes, as a share of the span
Y_PAD = 0.06


def price_range(low, high, x=None, window=None, pad: float = Y_PAD):
    """`[lo, hi]` for a price axis, fitted to the bars actually on screen.

    Plotly's autorange fits the whole *trace*, not the visible x-window, so a
    zoomed-in view kept the full session's height and the candles stayed flat —
    the zoom moved the time axis and nothing else. Computing the range from the
    rows inside the window is the only way to make the y-axis follow the
    movement, which is what a price chart is for.

    Each panel gets its own range: NIFTY's ~24,000 and a ₹120 premium share a
    time axis, never a price axis.

    `None` when there is nothing to fit, so the caller leaves autorange alone
    rather than pinning a bogus range.
    """
    lows = [_f(v) for v in _seq(low)]
    highs = [_f(v) for v in _seq(high)]
    if not lows or not highs:
        return None

    xs = _seq(x)
    if window and xs and len(xs) == len(lows):
        lo_hi = [(lows[i], highs[i]) for i in range(len(xs))
                 if _within(xs[i], window)]
        if lo_hi:
            lows = [a for a, _ in lo_hi]
            highs = [b for _, b in lo_hi]

    lo = min((v for v in lows if v is not None), default=None)
    hi = max((v for v in highs if v is not None), default=None)
    if lo is None or hi is None:
        return None
    span = hi - lo
    if span <= 0:
        # a dead-flat series still needs a visible band, or it renders as a
        # single line with no context at all
        band = abs(hi) * 0.001 or 1.0
        return [lo - band, hi + band]
    return [lo - span * pad, hi + span * pad]


def _within(v, window) -> bool:
    try:
        return window[0] <= v <= window[1]
    except (TypeError, IndexError):
        return True


#: how much of the price span the volume overlay is allowed to occupy
VOLUME_SHARE = 0.18


def _seq(v) -> List[Any]:
    """A plain list from a Series, array, list or `None`.

    Never `v or []`: a pandas Series has no unambiguous truth value and raises
    rather than falling back.
    """
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        return []


def volume_bars(volume, low, high, share: float = VOLUME_SHARE):
    """`(base, heights)` for the volume overlay, or `None` if it can't be drawn.

    A Plotly bar spans `base` → `base + y`. `y` is the bar's **length**, not the
    coordinate of its top. Passing the absolute top as `y` while also passing
    `base` made every bar reach `2 × low`, so the NIFTY panel auto-ranged to
    ~48,000 on a 24,000 index and the candles collapsed into a flat line at the
    bottom. The option legs flattened the same way, less visibly, because
    doubling a ₹120 premium's range is a smaller number but the same mistake.

    Heights are lengths measured up from `low.min()`, capped at `share` of the
    candle range, so the overlay always sits inside the price extent the candles
    already occupy and cannot influence autorange at all.
    """
    # NOT `volume or []`: these arrive as pandas Series, and `Series or []`
    # evaluates the Series for truthiness — "The truth value of a Series is
    # ambiguous" — which took the whole chart down, not just the overlay.
    vols = [_f(v) for v in _seq(volume)]
    lows = [_f(v) for v in _seq(low)]
    highs = [_f(v) for v in _seq(high)]
    finite_v = [v for v in vols if v is not None]
    finite_l = [v for v in lows if v is not None]
    finite_h = [v for v in highs if v is not None]
    if not finite_v or not finite_l or not finite_h:
        return None

    vmax = max(finite_v)
    if not vmax or vmax <= 0:
        return None
    base = min(finite_l)
    span = max(finite_h) - base
    if span <= 0:
        return None
    room = span * share
    return base, [0.0 if v is None else max(0.0, v) / vmax * room for v in vols]


def _add_signal(go, fig, nifty: Optional[Dict[str, Any]],
                signal: Optional[Dict[str, Any]]):
    """The decision state, pinned at the latest bar on the NIFTY panel."""
    s = dict(signal or {})
    state = str(s.get("state") or "").upper()
    mark = SIGNAL_MARKER.get(state)
    if not mark or nifty is None:
        return
    try:
        x = list(nifty["x"])[-1]
        y = float(list(nifty["close"])[-1])
    except Exception:
        return
    text, colour, symbol = mark
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text", showlegend=False,
        text=[text], textposition="top center",
        textfont=dict(size=10, color=colour),
        marker=dict(size=13, symbol=symbol, color=colour,
                    line=dict(width=1, color="#0b0f16")),
        hovertemplate=f"{text}<extra></extra>"), row=1, col=1)


def _tint(fig, tint: Optional[str], dominance: str):
    """A below-layer wash on the option panels only.

    Deliberately not `plot_bgcolor`: the candles have to stay readable, and a
    body colour that competes with them makes the chart harder to read rather
    than faster.
    """
    if not tint or dominance == "neutral":
        return
    for axis, yaxis in (("x2", "y2"), ("x3", "y3")):
        fig.add_shape(type="rect", xref=f"{axis} domain", yref=f"{yaxis} domain",
                      x0=0, x1=1, y0=0, y1=1, layer="below",
                      fillcolor=tint, line_width=0)


def atm_legs(leg_dfs: Optional[Dict[str, Any]]):
    """Find the ATM Call and ATM Put frames in the app's `_atm_leg_dfs` store.

    Keys look like `"ATM CE 23900"` / `"ATM+1 PE 24000"`. Exactly `ATM` wins;
    the nearest offset is a fallback so the terminal still draws something on a
    day the exact ATM leg failed to load.
    """
    legs = dict(leg_dfs or {})
    if not legs:
        return None, None, None, None

    def pick(side):
        exact = [k for k in legs if k.startswith("ATM ") and f" {side} " in k]
        if exact:
            return exact[0]
        near = sorted((k for k in legs if f" {side} " in k),
                      key=lambda k: abs(_offset(k)))
        return near[0] if near else None

    ce, pe = pick("CE"), pick("PE")
    return (legs.get(ce) if ce else None, legs.get(pe) if pe else None, ce, pe)


def _offset(tag: str) -> int:
    head = str(tag).split(" ")[0]
    if head == "ATM":
        return 0
    try:
        return int(head.replace("ATM", ""))
    except ValueError:
        return 99
