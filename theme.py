"""FoodOps.AI — Root Hub Theme & UI Design System.

Shares the same visual language as each submodule's theme.py (masthead,
section headers, badges, callouts) so navigating from the root console
into any module feels like one continuous product, even though every
module is an independently deployed Streamlit app.
"""

from typing import Any
import textwrap
import streamlit as st

ACCENT_COLOR = "#FF6B35"  # Warm Ember Orange — optimal visibility in both light & dark

CSS = textwrap.dedent("""
<style>
/* Global App Typography & Spacing adjustments */
.block-container {
    padding-top: 3.3rem;
    padding-bottom: 2.5rem;
    max-width: 1120px;
}

/* Masthead Header */
.fo-masthead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.2rem;
    padding-bottom: 1.1rem;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.22);
    flex-wrap: wrap;
}
.fo-brand-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.fo-wordmark {
    font-weight: 800;
    font-size: 1.65rem;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    color: inherit;
}
.fo-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #FF6B35;
    box-shadow: 0 0 10px rgba(255, 107, 53, 0.6);
    display: inline-block;
}
.fo-tagline {
    font-size: 0.90rem;
    color: inherit;
    opacity: 0.72;
}
.fo-masthead-meta {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.3rem;
}
.fo-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    background: rgba(16, 185, 129, 0.12);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.30);
}
@media (prefers-color-scheme: dark) {
    .fo-status-badge {
        background: rgba(34, 197, 94, 0.18);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.38);
    }
}
.fo-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}
.fo-meta-sub {
    font-size: 0.78rem;
    color: inherit;
    opacity: 0.65;
}

/* Section Header */
.fo-section {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 1.4rem 0 0.7rem 0;
}
.fo-section .fo-index {
    font-size: 0.75rem;
    font-weight: 700;
    color: #ea580c;
    background: rgba(234, 88, 12, 0.12);
    border: 1px solid rgba(234, 88, 12, 0.30);
    border-radius: 5px;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
@media (prefers-color-scheme: dark) {
    .fo-section .fo-index {
        color: #FF6B35;
        background: rgba(255, 107, 53, 0.16);
        border: 1px solid rgba(255, 107, 53, 0.36);
    }
}
.fo-section .fo-title {
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: -0.01em;
    color: inherit;
}
.fo-section .fo-rule {
    flex: 1;
    height: 1px;
    background: rgba(128, 128, 128, 0.22);
}
.fo-section-note {
    font-size: 0.86rem;
    color: inherit;
    opacity: 0.72;
    margin: -0.25rem 0 0.9rem 0;
    line-height: 1.45;
    max-width: 75ch;
}

/* Pricing / status Badges (shared badge language with submodules) */
.fo-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.32rem 0.75rem;
    border-radius: 6px;
    margin-top: 0.25rem;
    margin-bottom: 0.4rem;
}
.fo-badge.discount {
    background: rgba(16, 185, 129, 0.12);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.32);
}
@media (prefers-color-scheme: dark) {
    .fo-badge.discount {
        background: rgba(34, 197, 94, 0.18);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.40);
    }
}
.fo-badge.markup {
    background: rgba(239, 68, 68, 0.12);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.30);
}
@media (prefers-color-scheme: dark) {
    .fo-badge.markup {
        background: rgba(239, 68, 68, 0.20);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.42);
    }
}
.fo-badge.neutral {
    background: rgba(128, 128, 128, 0.10);
    color: inherit;
    border: 1px solid rgba(128, 128, 128, 0.24);
}
.fo-swatch {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}

/* Callout Alert Box */
.fo-callout {
    background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-left: 3.5px solid #3b82f6;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.88rem;
    line-height: 1.45;
    color: inherit;
    margin: 0.7rem 0 1.1rem 0;
}

/* Module Cards */
.fo-card {
    background: var(--secondary-background-color, rgba(128, 128, 128, 0.05));
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 12px;
    padding: 1.15rem 1.3rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.fo-card:hover {
    border-color: rgba(255, 107, 53, 0.45);
    transform: translateY(-2px);
}
.fo-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
}
.fo-card-name {
    font-size: 1.08rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: inherit;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.fo-card-tagline {
    font-size: 0.85rem;
    color: inherit;
    opacity: 0.72;
    margin-top: -0.35rem;
}
.fo-card-desc {
    font-size: 0.85rem;
    color: inherit;
    opacity: 0.86;
    line-height: 1.45;
    min-height: 3.4em;
}
.fo-card-metrics {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.1rem;
}
.fo-metric-chip {
    font-size: 0.76rem;
    font-weight: 600;
    padding: 0.22rem 0.6rem;
    border-radius: 6px;
    background: rgba(255, 107, 53, 0.10);
    color: #ea580c;
    border: 1px solid rgba(255, 107, 53, 0.26);
}
@media (prefers-color-scheme: dark) {
    .fo-metric-chip {
        color: #FF6B35;
        background: rgba(255, 107, 53, 0.16);
        border: 1px solid rgba(255, 107, 53, 0.36);
    }
}
.fo-tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.15rem;
}
.fo-tag-chip {
    font-size: 0.72rem;
    padding: 0.14rem 0.55rem;
    border-radius: 6px;
    background: rgba(99, 102, 241, 0.10);
    color: #6366f1;
    border: 1px solid rgba(99, 102, 241, 0.22);
}
@media (prefers-color-scheme: dark) {
    .fo-tag-chip {
        background: rgba(129, 140, 248, 0.16);
        color: #a5b4fc;
        border: 1px solid rgba(129, 140, 248, 0.32);
    }
}

/* Stat strip (root-level KPI summary) */
.fo-stat {
    text-align: center;
    padding: 0.9rem 0.4rem;
}
.fo-stat-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: inherit;
    letter-spacing: -0.02em;
}
.fo-stat-label {
    font-size: 0.78rem;
    color: inherit;
    opacity: 0.65;
    margin-top: 0.15rem;
}
</style>
""").strip()


def apply_theme() -> None:
    """Inject dynamic Light/Dark CSS rules into Streamlit DOM."""
    st.markdown(CSS, unsafe_allow_html=True)


def masthead(
    week_label: str = "4 Independent Modules · Real-World & Simulated Operational Data",
    model_label: str = "Platform Status: 2 Live · 2 Queued",
) -> None:
    """Render executive top navigation masthead for the root hub."""
    html = textwrap.dedent(f"""\
<div class="fo-masthead">
    <div class="fo-brand-group">
        <div class="fo-wordmark"><span class="fo-dot"></span>FoodOps.AI</div>
        <div class="fo-tagline">Enterprise Food Delivery Operational Intelligence Platform</div>
    </div>
    <div class="fo-masthead-meta">
        <div class="fo-status-badge"><span class="fo-status-dot"></span>{model_label}</div>
        <div class="fo-meta-sub">{week_label}</div>
    </div>
</div>
""").strip()
    st.markdown(html, unsafe_allow_html=True)


def section(index: str, title: str, note: str = "") -> None:
    """Render professional section divider with index pill."""
    html = textwrap.dedent(f"""\
<div class="fo-section">
    <div class="fo-index">{index}</div>
    <div class="fo-title">{title}</div>
    <div class="fo-rule"></div>
</div>
""").strip()
    st.markdown(html, unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="fo-section-note">{note}</div>', unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Render a dynamic live/queued pill badge, reusing the discount/markup/neutral language."""
    if status == "live":
        return '<div class="fo-badge discount"><span class="fo-swatch"></span>Live &amp; Deployed</div>'
    return '<div class="fo-badge neutral"><span class="fo-swatch"></span>Queued</div>'


def callout(text: str) -> None:
    """Render an informational callout container."""
    st.markdown(f'<div class="fo-callout">{text}</div>', unsafe_allow_html=True)


def stat(value: str, label: str) -> str:
    """Render a single KPI stat block for the platform-level summary strip."""
    return textwrap.dedent(f"""\
<div class="fo-stat">
    <div class="fo-stat-value">{value}</div>
    <div class="fo-stat-label">{label}</div>
</div>
""").strip()


def module_card(
    icon: str,
    name: str,
    tagline: str,
    description: str,
    status: str,
    metrics: list,
    tags: list,
) -> None:
    """Render a full module console card (header, tagline, description, metric chips, tags)."""
    metrics_html = "".join(
        f'<span class="fo-metric-chip">{m}</span>' for m in metrics
    )
    tags_html = "".join(f'<span class="fo-tag-chip">{t}</span>' for t in tags)

    html = textwrap.dedent(f"""\
<div class="fo-card">
    <div class="fo-card-header">
        <div class="fo-card-name">{icon} {name}</div>
    </div>
    <div class="fo-card-tagline">{tagline}</div>
    {status_badge(status)}
    <div class="fo-card-desc">{description}</div>
    <div class="fo-card-metrics">{metrics_html}</div>
    <div class="fo-tag-row">{tags_html}</div>
</div>
""").strip()
    st.markdown(html, unsafe_allow_html=True)


def chart_layout(fig: Any, **kwargs) -> Any:
    """Apply theme-neutral styling to Plotly figures.

    Avoids hardcoding text or background colors, allowing Streamlit's native
    Plotly theme engine to dynamically render clean fonts in both Light and Dark modes.
    """
    xaxis_extra = kwargs.pop("xaxis", {})
    yaxis_extra = kwargs.pop("yaxis", {})

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=24, t=46, b=24),
        xaxis={
            **dict(
                gridcolor="rgba(128, 128, 128, 0.18)",
                zerolinecolor="rgba(128, 128, 128, 0.26)",
            ),
            **xaxis_extra
        },
        yaxis={
            **dict(
                gridcolor="rgba(128, 128, 128, 0.18)",
                zerolinecolor="rgba(128, 128, 128, 0.26)",
            ),
            **yaxis_extra
        },
        hoverlabel=dict(
            font_size=12,
        ),
        **kwargs,
    )
    return fig