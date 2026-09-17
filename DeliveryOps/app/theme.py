"""FoodOps.AI — Theme & UI Design System.

Designed to dynamically harmonize with Streamlit's built-in Light and Dark themes
using native CSS custom properties and adaptive styling.
Guarantees high contrast, professional typography, and responsive readability.
"""

from typing import List, Optional, Any
import textwrap
import streamlit as st

ACCENT_COLOR = "#FF6B35"  # Warm Ember Orange — optimal visibility in both light & dark

CSS = textwrap.dedent("""
<style>
/* Global App Typography & Spacing adjustments */
.block-container {
    padding-top: 3.3rem;
    padding-bottom: 2.5rem;
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

/* Pricing Badges */
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
</style>
""").strip()


def apply_theme() -> None:
    """Inject dynamic Light/Dark CSS rules into Streamlit DOM."""
    st.markdown(CSS, unsafe_allow_html=True)


def masthead(week_label: str = "Historical panel: Jan - Feb 2015 · 6 Markets · ~196k Orders", model_label: str = "LightGBM Quantile Regressor (10.29m MAE)") -> None:
    """Render executive top navigation masthead."""
    html = textwrap.dedent(f"""\
<div class="fo-masthead">
    <div class="fo-brand-group">
        <div class="fo-wordmark"><span class="fo-dot"></span>FoodOps.AI [DeliveryOps]</div>
        <div class="fo-tagline">Decentralized Delivery &amp; Scenario Simulator</div>
    </div>
    <div class="fo-masthead-meta">
        <div class="fo-status-badge"><span class="fo-status-dot"></span>Model: {model_label}</div>
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


def price_badge(price_change_pct: float) -> str:
    """Render a dynamic discount/markup pill badge."""
    if price_change_pct < -0.001:
        return f'<div class="fo-badge discount"><span class="fo-swatch"></span>Discount: {abs(price_change_pct)*100:.1f}% below catalog base</div>'
    if price_change_pct > 0.001:
        return f'<div class="fo-badge markup"><span class="fo-swatch"></span>Markup: +{price_change_pct*100:.1f}% above catalog base</div>'
    return '<div class="fo-badge neutral"><span class="fo-swatch"></span>Standard Catalog Price (0.0% delta)</div>'


def callout(text: str) -> None:
    """Render an informational callout container."""
    st.markdown(f'<div class="fo-callout">{text}</div>', unsafe_allow_html=True)


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
