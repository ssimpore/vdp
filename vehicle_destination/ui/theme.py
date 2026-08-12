"""Reference-faithful visual system for the Streamlit research application."""

from __future__ import annotations


APP_CSS = r"""
<style>
:root {
  --vdp-navy-950: #031f36;
  --vdp-navy-900: #062944;
  --vdp-navy-800: #0d3d68;
  --vdp-blue-700: #0759b6;
  --vdp-blue-600: #0968d8;
  --vdp-blue-100: #e7f1ff;
  --vdp-ink: #10233b;
  --vdp-text: #27364a;
  --vdp-muted: #667085;
  --vdp-subtle: #8b97a8;
  --vdp-line: #d7dde5;
  --vdp-line-soft: #e9edf2;
  --vdp-surface: #ffffff;
  --vdp-surface-soft: #f7f9fb;
  --vdp-green: #14823b;
  --vdp-red: #d53030;
  --vdp-radius: 7px;
  --vdp-control-height: 40px;
  --vdp-sidebar-width: 215px;
  --vdp-shadow: 0 1px 2px rgba(16, 35, 59, .04);
  --vdp-font: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

html, body, [class*="css"] { font-family: var(--vdp-font); }
html { font-size: 14px; }
.stApp { background: var(--vdp-surface); color: var(--vdp-text); }
.stAppViewContainer { background: var(--vdp-surface); }
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
  max-width: 1600px;
  padding: 0 16px 24px 28px;
}
[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
  width: var(--vdp-sidebar-width) !important;
  min-width: var(--vdp-sidebar-width) !important;
  background: linear-gradient(180deg, var(--vdp-navy-950) 0%, var(--vdp-navy-900) 100%);
  border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] > div:first-child {
  width: var(--vdp-sidebar-width) !important;
  padding: 0 8px 12px;
}
[data-testid="stSidebarContent"] { padding: 0 8px 12px !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 5px; min-height: calc(100vh - 20px); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.vdp-sidebar-footer) { margin-top: auto; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.13); margin: 18px 0; }
[data-testid="stSidebar"] .stButton { margin: 0; }
[data-testid="stSidebar"] .stButton > button {
  min-height: 48px;
  width: 100%;
  justify-content: flex-start;
  gap: 11px;
  padding: 0 13px;
  color: #f8fbff !important;
  background: transparent;
  border: 0;
  border-radius: 7px;
  box-shadow: none;
  font-size: 14px;
  font-weight: 450;
}
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button svg { color: inherit !important; fill: currentColor; }
[data-testid="stSidebar"] .stButton > button:hover {
  color: #ffffff;
  background: rgba(42, 105, 170, .35);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  color: #ffffff;
  background: linear-gradient(90deg, #164d84 0%, #1a4e86 100%);
  font-weight: 650;
}
[data-testid="stSidebar"] .stButton > button:focus-visible,
button:focus-visible, input:focus-visible, [role="combobox"]:focus-visible {
  outline: 3px solid rgba(88, 166, 255, .45) !important;
  outline-offset: 2px;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] { display: none; }

.vdp-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  min-height: 104px;
  padding: 18px 12px 17px 10px;
  color: #ffffff;
  margin-top: -54px;
}
.vdp-brand__mark {
  width: 31px;
  height: 44px;
  flex: 0 0 auto;
  filter: drop-shadow(0 2px 5px rgba(0,0,0,.18));
}
.vdp-brand__name { font-size: 16px; font-weight: 720; line-height: 1.3; letter-spacing: -.015em; }
.vdp-brand__meta { display: none; }
.vdp-sidebar-footer {
  position: fixed;
  left: 18px;
  bottom: 24px;
  width: 179px;
  margin-top: 24px;
  padding: 16px 13px 0;
  border-top: 1px solid rgba(255,255,255,.13);
  color: #b7c9d9;
  font-size: 11px;
  line-height: 1.55;
}
.vdp-sidebar-footer strong { color: #ffffff; font-weight: 600; }

/* Page typography */
h1, h2, h3, h4, p { font-family: var(--vdp-font); }
h1, h2, h3, h4 { color: var(--vdp-ink); letter-spacing: -.026em; }
h1 { font-size: 32px !important; line-height: 1.12 !important; font-weight: 760 !important; margin: 0 0 2px !important; }
h2 { font-size: 19px !important; line-height: 1.3 !important; font-weight: 700 !important; margin-top: 14px !important; }
h3 { font-size: 15px !important; font-weight: 700 !important; }
p, label, .stCaption { color: var(--vdp-text); }
.vdp-page-header { margin: -10px 0 22px; }
.vdp-page-title { color: var(--vdp-ink); font-size: 32px; line-height: 1.12; font-weight: 760; letter-spacing: -.035em; }
.vdp-page-subtitle { margin-top: 3px; color: #5f6876; font-size: 14px; line-height: 1.45; }

/* Inputs and actions */
[data-testid="stWidgetLabel"] p {
  color: var(--vdp-ink);
  font-size: 12px;
  line-height: 1.2;
  font-weight: 660;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] [data-baseweb="input"] > div,
[data-testid="stTextInput"] [data-baseweb="input"] > div,
[data-testid="stDateInput"] [data-baseweb="input"] > div,
[data-testid="stFileUploaderDropzone"] {
  min-height: var(--vdp-control-height);
  background: #ffffff;
  border-color: #cbd3dd;
  border-radius: var(--vdp-radius);
  box-shadow: var(--vdp-shadow);
}
[data-testid="stSelectbox"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input { font-size: 12px !important; }
[data-baseweb="popover"] { font-family: var(--vdp-font); }
.stButton > button, .stDownloadButton > button {
  min-height: var(--vdp-control-height);
  border-radius: var(--vdp-radius);
  border: 1px solid #cbd3dd;
  padding: 0 16px;
  font-size: 12px;
  font-weight: 650;
  box-shadow: var(--vdp-shadow);
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  color: #ffffff;
  background: linear-gradient(180deg, #0b6bd7 0%, #075bbb 100%);
  border-color: #075bbb;
  box-shadow: 0 3px 8px rgba(7, 91, 187, .16);
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stDownloadButton > button[kind="primary"] p,
.stDownloadButton > button[kind="primary"] span { color: #ffffff !important; }
.stButton > button[kind="primary"]:hover { background: #064f9f; border-color: #064f9f; }
[data-testid="stCheckbox"] label p { color: var(--vdp-text); font-size: 11px; font-weight: 500; }
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child { border-radius: 3px; }
[data-testid="stCheckbox"] [aria-checked="true"] > div:first-child,
[data-testid="stCheckbox"] input:checked + div { background-color: var(--vdp-blue-600) !important; border-color: var(--vdp-blue-600) !important; }

/* Panels, metrics and tables */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--vdp-line) !important;
  border-radius: var(--vdp-radius) !important;
  box-shadow: var(--vdp-shadow);
}
[data-testid="stMetric"] {
  min-height: 84px;
  padding: 13px 15px;
  border: 1px solid var(--vdp-line);
  border-radius: var(--vdp-radius);
  background: #ffffff;
  box-shadow: var(--vdp-shadow);
}
[data-testid="stMetricLabel"] p { color: var(--vdp-muted); font-size: 11px; }
[data-testid="stMetricValue"] { color: var(--vdp-ink); font-size: 22px; font-weight: 720; }
[data-testid="stDataFrame"] {
  overflow: hidden;
  border: 1px solid var(--vdp-line);
  border-radius: var(--vdp-radius);
  box-shadow: var(--vdp-shadow);
}
[data-testid="stDataFrame"] * { font-size: 11.5px; }
[data-testid="stDataFrame"] [role="columnheader"] { color: var(--vdp-ink); font-weight: 700; }
[data-testid="stAlert"] { border-radius: var(--vdp-radius); font-size: 12px; }
[data-testid="stTabs"] [role="tablist"] { gap: 22px; border-bottom: 1px solid var(--vdp-line-soft); }
[data-testid="stTabs"] button[role="tab"] { padding: 7px 0 10px; font-size: 12px; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--vdp-blue-700); }
[data-testid="stExpander"] { border-color: var(--vdp-line); border-radius: var(--vdp-radius); }
[data-testid="stJson"] { max-height: 460px; }

.vdp-section-title { margin: 3px 0 11px; color: var(--vdp-ink); font-size: 13px; font-weight: 700; }
.vdp-panel-title { margin-bottom: 13px; color: var(--vdp-ink); font-size: 12px; font-weight: 720; }
.vdp-kpi-stack { border: 1px solid var(--vdp-line); border-radius: var(--vdp-radius); background: #fff; }
.vdp-kpi-row { display: grid; grid-template-columns: 1fr auto; gap: 14px; align-items: center; min-height: 60px; padding: 12px 16px; border-bottom: 1px solid var(--vdp-line-soft); }
.vdp-kpi-row:last-child { border-bottom: 0; }
.vdp-kpi-label { color: var(--vdp-text); font-size: 11.5px; }
.vdp-kpi-value { color: var(--vdp-ink); font-size: 13px; font-weight: 720; text-align: right; }
.vdp-kpi-value--good { color: var(--vdp-green); }
.vdp-kpi-value--bad { color: var(--vdp-red); }
.vdp-detail-panel { padding: 14px 15px 8px; border: 1px solid var(--vdp-line); border-radius: var(--vdp-radius); background: #fff; min-height: 444px; }
.vdp-detail-group { padding: 0 0 11px; margin: 0 0 12px; border-bottom: 1px solid var(--vdp-line-soft); }
.vdp-detail-group:last-child { border-bottom: 0; margin-bottom: 0; }
.vdp-detail-label { margin-top: 9px; color: var(--vdp-muted); font-size: 10.5px; }
.vdp-detail-value { margin-top: 1px; color: var(--vdp-text); font-size: 11.5px; line-height: 1.45; overflow-wrap: anywhere; }
.vdp-status-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.vdp-status-table th { padding: 9px 12px; color: var(--vdp-muted); font-size: 10.5px; font-weight: 700; text-align: left; border-bottom: 1px solid var(--vdp-line); text-transform: uppercase; letter-spacing: .035em; }
.vdp-status-table td { padding: 11px 12px; border-bottom: 1px solid var(--vdp-line-soft); }
.vdp-status-dot { display: inline-block; width: 7px; height: 7px; margin-right: 7px; border-radius: 50%; background: var(--vdp-green); }
.vdp-status-dot--muted { background: #a8b2c0; }
.vdp-science-note { margin-top: 14px; padding: 12px 14px; border-left: 3px solid var(--vdp-blue-600); background: #f3f7fc; color: #42536a; font-size: 11.5px; line-height: 1.5; }
.vdp-run-footer { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 10px; color: var(--vdp-muted); font-size: 10.5px; }
.vdp-run-footer__separator { color: #bac2cc; }

/* Map framing */
[data-testid="stDeckGlJsonChart"], [data-testid="stIFrame"] {
  overflow: hidden;
  border: 1px solid var(--vdp-line);
  border-radius: var(--vdp-radius);
  background: #eef3f6;
  box-shadow: var(--vdp-shadow);
}
[data-testid="stIFrame"] iframe { display: block; }
.vdp-map-legend { display: flex; flex-wrap: wrap; gap: 8px 17px; padding: 8px 11px; margin-top: -4px; border: 1px solid var(--vdp-line); border-top: 0; border-radius: 0 0 var(--vdp-radius) var(--vdp-radius); background: #fff; color: var(--vdp-text); font-size: 10.5px; }
.vdp-legend-item { display: inline-flex; gap: 6px; align-items: center; }
.vdp-legend-dot { width: 8px; height: 8px; border-radius: 50%; border: 2px solid currentColor; background: #fff; }
.vdp-legend-line { width: 15px; height: 2px; background: currentColor; }

@media (max-width: 1000px) {
  [data-testid="stAppViewBlockContainer"], [data-testid="stMainBlockContainer"] { padding: 20px 18px 28px; }
  h1, .vdp-page-title { font-size: 27px !important; }
  .vdp-detail-panel { min-height: auto; }
}
@media (max-width: 700px) {
  [data-testid="stAppViewBlockContainer"], [data-testid="stMainBlockContainer"] { padding: 18px 14px 24px; }
  .vdp-page-header { margin-bottom: 18px; }
  h1, .vdp-page-title { font-size: 25px !important; }
  .vdp-page-subtitle { font-size: 12.5px; }
  [data-testid="stMetric"] { min-height: 74px; }
  .vdp-kpi-row { min-height: 52px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
}
</style>
"""


BRAND_HTML = r"""
<div class="vdp-brand" aria-label="Vehicle Destination Lab">
  <svg class="vdp-brand__mark" viewBox="0 0 40 54" role="img" aria-hidden="true">
    <defs>
      <linearGradient id="vdp-liquid" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#6de0f1"/><stop offset="1" stop-color="#39bd79"/>
      </linearGradient>
    </defs>
    <path d="M14 3h12M17 3v15L7.8 34.2C2.7 43.1 9.1 51 20 51s17.3-7.9 12.2-16.8L23 18V3" fill="none" stroke="#d9f1ff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M9.2 36.6c4.1-1.9 7.1 1.5 11 .3 3.8-1.2 5.3-4.2 10.5-1.8 3.6 7.1-1.7 12.6-10.7 12.6-8.7 0-14.2-5-10.8-11.1Z" fill="url(#vdp-liquid)"/>
    <circle cx="15" cy="34" r="1.7" fill="#e8fff4"/><circle cx="25.5" cy="40" r="1.2" fill="#e8fff4"/>
  </svg>
  <div>
    <div class="vdp-brand__name">Vehicle<br/>Destination Lab</div>
    <div class="vdp-brand__meta">Research workspace</div>
  </div>
</div>
"""


def inject_theme(st) -> None:
    """Install the application design tokens and reference-faithful CSS."""

    st.markdown(APP_CSS, unsafe_allow_html=True)
