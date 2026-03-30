import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

from optimizers import beale_numpy, run

# --- Surface mesh (precomputed, log1p-compressed) ---
_lin = np.linspace(-5, 5, 80)
X, Y = np.meshgrid(_lin, _lin)
Z = np.log1p(beale_numpy(X, Y))

SURFACE = go.Surface(
    x=_lin, y=_lin, z=Z,
    colorscale='Viridis', opacity=0.7,
    showscale=False, name='Beale (log1p)',
    hoverinfo='skip',
)

COLORS = {
    'sgd':      '#EF4444',
    'momentum': '#3B82F6',
    'nesterov': '#10B981',
    'adam':     '#F59E0B',
}

LAYOUT = go.Layout(
    scene=dict(
        xaxis_title='w₁', yaxis_title='w₂', zaxis_title='log(J+1)',
        xaxis=dict(backgroundcolor='#0B1120', gridcolor='#1E3050', zerolinecolor='#1E3050'),
        yaxis=dict(backgroundcolor='#0B1120', gridcolor='#1E3050', zerolinecolor='#1E3050'),
        zaxis=dict(backgroundcolor='#0B1120', gridcolor='#1E3050', zerolinecolor='#1E3050'),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        aspectmode='cube',
    ),
    paper_bgcolor='#0B1120',
    plot_bgcolor='#0B1120',
    font=dict(color='#94A3B8', family='Consolas, monospace'),
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
        x=0.01, y=0.99,
        bgcolor='rgba(11,17,32,0.85)',
        bordercolor='#1E3050',
        borderwidth=1,
        font=dict(size=12, color='#E2E8F0'),
    ),
)


def build_figure(lr, mu, b1, b2, n, selected):
    traces = [SURFACE]
    for opt in (selected or []):
        xs, ys, js = run(opt, lr, mu, b1, b2, n)
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=js,
            mode='lines+markers',
            line=dict(color=COLORS[opt], width=4),
            marker=dict(
                size=[2] * (len(xs) - 1) + [10],
                color=[COLORS[opt]] * len(xs),
                symbol=['circle'] * (len(xs) - 1) + ['diamond'],
            ),
            name=opt.upper(),
        ))
    return go.Figure(data=traces, layout=LAYOUT)


DEFAULTS = dict(lr=0.01, mu=0.9, b1=0.9, b2=0.999, n=200,
                selected=['sgd', 'momentum', 'nesterov', 'adam'])
INITIAL_FIG = build_figure(**DEFAULTS)

# ── CSS injected into <head> ───────────────────────────────────────────────
GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --navy:   #0B1120;
  --panel:  #111827;
  --card:   #131D30;
  --border: #1E3050;
  --teal:   #0EA5E9;
  --green:  #10B981;
  --red:    #EF4444;
  --amber:  #F59E0B;
  --blue:   #3B82F6;
  --white:  #F1F5F9;
  --muted:  #64748B;
  --light:  #94A3B8;
}

body { background: var(--navy); color: var(--white); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Keyframes ── */
@keyframes panelSlideIn {
  from { opacity: 0; transform: translateX(-18px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes plotFadeIn {
  from { opacity: 0; transform: scale(0.985); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes dotPulse {
  0%, 100% { box-shadow: 0 0 4px 1px currentColor; }
  50%       { box-shadow: 0 0 10px 3px currentColor; }
}
@keyframes headerGlow {
  0%, 100% { text-shadow: 0 0 8px rgba(14,165,233,0.3); }
  50%       { text-shadow: 0 0 20px rgba(14,165,233,0.7), 0 0 40px rgba(14,165,233,0.2); }
}
@keyframes scanline {
  from { background-position: 0 0; }
  to   { background-position: 0 100%; }
}
@keyframes borderPulse {
  0%, 100% { border-left-color: var(--teal); }
  50%       { border-left-color: rgba(14,165,233,0.3); }
}

/* ── Panel slide-in on load ── */
#left-panel {
  animation: panelSlideIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* ── Plot fade-in on load ── */
#right-panel {
  animation: plotFadeIn 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.15s both;
}

/* ── Header icon glow ── */
.header-icon {
  animation: headerGlow 3s ease-in-out infinite;
  display: inline-block;
}

/* ── Staggered fade-up for panel children ── */
.fade-item { opacity: 0; animation: fadeUp 0.45s ease forwards; }
.fade-item:nth-child(1) { animation-delay: 0.10s; }
.fade-item:nth-child(2) { animation-delay: 0.18s; }
.fade-item:nth-child(3) { animation-delay: 0.26s; }
.fade-item:nth-child(4) { animation-delay: 0.34s; }
.fade-item:nth-child(5) { animation-delay: 0.42s; }
.fade-item:nth-child(6) { animation-delay: 0.50s; }
.fade-item:nth-child(7) { animation-delay: 0.58s; }

/* ── Optimizer toggle cards: hover lift ── */
.opt-card {
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
  cursor: default;
}
.opt-card:hover {
  transform: translateX(3px);
  box-shadow: -3px 0 12px rgba(0,0,0,0.4);
}

/* ── Hyperparameter cards: hover glow ── */
.param-card {
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.param-card:hover {
  box-shadow: 0 0 16px rgba(14,165,233,0.08), inset 0 0 20px rgba(14,165,233,0.02);
}

/* ── Dot pulse on each optimizer ── */
.opt-dot {
  animation: dotPulse 2.5s ease-in-out infinite;
}
.opt-dot-sgd      { color: #EF4444; animation-delay: 0.0s; }
.opt-dot-momentum { color: #3B82F6; animation-delay: 0.6s; }
.opt-dot-nesterov { color: #10B981; animation-delay: 1.2s; }
.opt-dot-adam     { color: #F59E0B; animation-delay: 1.8s; }

/* ── Top bar scan line ── */
.top-bar {
  position: relative;
  overflow: hidden;
}
.top-bar::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    transparent 0%, rgba(14,165,233,0.03) 50%, transparent 100%
  );
  background-size: 100% 40px;
  animation: scanline 4s linear infinite;
  pointer-events: none;
}

/* ── Slider overrides ── */
.rc-slider-rail  { background: var(--border) !important; height: 3px !important; }
.rc-slider-track { background: var(--teal)   !important; height: 3px !important;
                   transition: width 0.12s ease !important; }
.rc-slider-handle {
  border-color: var(--teal) !important;
  background: var(--navy)  !important;
  box-shadow: 0 0 0 2px var(--teal) !important;
  width: 14px !important; height: 14px !important;
  margin-top: -5px !important;
  transition: box-shadow 0.15s ease !important;
}
.rc-slider-handle:hover, .rc-slider-handle-click-focused {
  box-shadow: 0 0 0 5px rgba(14,165,233,0.22) !important;
}
.rc-slider-tooltip-inner {
  background: #131D30 !important;
  color: #0EA5E9 !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  border: 1px solid #1E3050 !important;
  border-radius: 4px !important;
  padding: 2px 8px !important;
  box-shadow: none !important;
}

/* ── Checkbox ── */
input[type="checkbox"] {
  appearance: none; -webkit-appearance: none;
  width: 15px; height: 15px;
  border: 1.5px solid #1E3050;
  border-radius: 3px;
  background: #0B1120;
  cursor: pointer; flex-shrink: 0;
  transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
  position: relative;
}
input[type="checkbox"]:hover { border-color: var(--teal); box-shadow: 0 0 0 3px rgba(14,165,233,0.15); }
input[type="checkbox"]:checked { background: #0EA5E9; border-color: #0EA5E9; }
input[type="checkbox"]:checked::after {
  content: '';
  position: absolute;
  left: 3px; top: 1px;
  width: 5px; height: 8px;
  border: 2px solid white;
  border-top: none; border-left: none;
  transform: rotate(45deg);
}
"""

# ── Layout helpers ─────────────────────────────────────────────────────────

def card(children, accent=None, extra_style=None):
    base = {
        'background': '#131D30',
        'border': '1px solid ' + (accent + '44' if accent else '#1E3050'),
        'borderLeft': '3px solid ' + (accent if accent else '#1E3050'),
        'borderRadius': '6px',
        'padding': '14px 16px',
        'marginBottom': '12px',
    }
    if extra_style:
        base.update(extra_style)
    return html.Div(children, style=base)


def label_text(text):
    return html.P(text, style={
        'fontFamily': "'IBM Plex Sans', sans-serif",
        'fontSize': '10px',
        'fontWeight': '600',
        'color': '#94A3B8',
        'letterSpacing': '0.1em',
        'textTransform': 'uppercase',
        'marginBottom': '8px',
    })


def info_text(text, color='#475569'):
    return html.P(text, style={
        'fontFamily': "'IBM Plex Sans', sans-serif",
        'fontSize': '11px',
        'color': color,
        'lineHeight': '1.6',
        'marginTop': '6px',
    })


def section_header(icon, title):
    return html.Div([
        html.Span(icon, style={'fontSize': '12px', 'marginRight': '7px', 'color': '#0EA5E9'}),
        html.Span(title, style={
            'fontFamily': "'IBM Plex Mono', monospace",
            'fontSize': '10px',
            'fontWeight': '600',
            'letterSpacing': '0.14em',
            'textTransform': 'uppercase',
            'color': '#0EA5E9',
        }),
    ], style={'marginBottom': '10px', 'marginTop': '8px'})


def slider_block(slider_id, label, description, **slider_kwargs):
    return html.Div([
        label_text(label),
        dcc.Slider(id=slider_id, marks=None,
                   tooltip={"placement": "bottom", "always_visible": False},
                   **slider_kwargs),
        info_text(description),
    ], style={'marginBottom': '16px'})


OPT_META = {
    'sgd':      ('#EF4444', 'SGD',      'Descenso puro. Sin memoria. Errático en valles curvos.'),
    'momentum': ('#3B82F6', 'Momentum', 'SGD + inercia acumulada. Suaviza oscilaciones.'),
    'nesterov': ('#10B981', 'Nesterov', 'Momentum con gradiente anticipado. Corrección más precisa.'),
    'adam':     ('#F59E0B', 'ADAM',     'Lr adaptativo por parámetro. Robusto y popular.'),
}


def optimizer_toggle(key, color, name, desc):
    return html.Div([
        html.Div([
            html.Div(className=f'opt-dot opt-dot-{key}', style={
                'width': '9px', 'height': '9px', 'borderRadius': '50%',
                'background': color, 'flexShrink': '0',
                'color': color,   # used by currentColor in CSS animation
                'marginTop': '2px',
            }),
            html.Div([
                html.Span(name, style={
                    'fontFamily': "'IBM Plex Mono', monospace",
                    'fontSize': '12px', 'fontWeight': '600', 'color': color,
                    'display': 'block',
                }),
                html.Span(desc, style={
                    'fontFamily': "'IBM Plex Sans', sans-serif",
                    'fontSize': '10px', 'color': '#475569',
                    'display': 'block', 'marginTop': '1px',
                }),
            ], style={'flex': '1'}),
            dcc.Checklist(
                id=f'opt-{key}',
                options=[{'label': '', 'value': key}],
                value=[key],
                style={'display': 'flex', 'alignItems': 'center'},
                inputStyle={'cursor': 'pointer'},
            ),
        ], style={'display': 'flex', 'alignItems': 'flex-start', 'gap': '10px'}),
    ], className='opt-card', style={
        'background': '#0D1829',
        'border': f'1px solid {color}22',
        'borderLeft': f'3px solid {color}',
        'borderRadius': '5px',
        'padding': '10px 12px',
        'marginBottom': '7px',
    })


# ── App ────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.index_string = f'''<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>Optimizer Physics</title>
    {{%favicon%}}
    {{%css%}}
    <style>{GLOBAL_CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>'''

app.layout = html.Div(style={
    'display': 'flex',
    'background': '#0B1120',
    'height': '100vh',
    'fontFamily': "'IBM Plex Sans', sans-serif",
    'overflow': 'hidden',
}, children=[

    # ── LEFT PANEL ────────────────────────────────────────────────────────
    html.Div(id='left-panel', style={
        'width': '290px',
        'minWidth': '290px',
        'background': '#111827',
        'borderRight': '1px solid #1E3050',
        'display': 'flex',
        'flexDirection': 'column',
        'overflowY': 'auto',
        'overflowX': 'hidden',
    }, children=[

        # ── Header ──
        html.Div([
            html.Div([
                html.Span("◈ ", className='header-icon', style={'color': '#0EA5E9', 'fontSize': '16px'}),
                html.Span("Optimizer Physics", style={
                    'fontFamily': "'IBM Plex Mono', monospace",
                    'fontSize': '14px', 'fontWeight': '600', 'color': '#F1F5F9',
                }),
            ], style={'marginBottom': '8px'}),
            info_text(
                "Visualización 3D del descenso sobre la función de Beale — "
                "benchmark clásico con mínimo global en (3, 0.5).",
                color='#334155',
            ),
        ], style={
            'padding': '18px 18px 14px',
            'borderBottom': '1px solid #1E3050',
            'background': 'linear-gradient(180deg, #131D30 0%, #111827 100%)',
        }),

        # ── Body ──
        html.Div(style={'padding': '14px 16px', 'flex': '1'}, children=[

            html.Div(className='fade-item', children=[section_header('⚙', 'Hiperparámetros Globales')]),

            html.Div(className='fade-item', children=[card([
                slider_block('lr', 'Learning Rate  η',
                    'Tamaño del paso. Alto = rápido pero inestable. Bajo = lento pero seguro.',
                    min=0.001, max=0.1, step=0.001, value=0.01),
                slider_block('mu', 'Momentum  μ',
                    'Fracción de velocidad conservada. μ=0 es SGD puro; alto = más inercia.',
                    min=0.0, max=0.99, step=0.01, value=0.9),
            ], accent='#0EA5E9')]),

            html.Div(className='fade-item', children=[card([
                slider_block('beta1', 'β₁ — Primer momento',
                    'Media móvil del gradiente en ADAM. Controla el impulso. Típico: 0.9.',
                    min=0.8, max=0.99, step=0.01, value=0.9),
                slider_block('beta2', 'β₂ — Segundo momento',
                    'Media móvil del gradiente². Escala el lr por coordenada. Típico: 0.999.',
                    min=0.9, max=0.999, step=0.001, value=0.999),
            ], accent='#F59E0B')]),

            html.Div(className='fade-item', children=[card([
                slider_block('iters', 'Iteraciones',
                    'Número de pasos de descenso. Más pasos revelan convergencia o divergencia.',
                    min=50, max=500, step=50, value=200),
            ], accent='#10B981')]),

            html.Div(className='fade-item', children=[section_header('◉', 'Optimizadores activos')]),

            html.Div(className='fade-item', children=[
                *[optimizer_toggle(k, c, n, d) for k, (c, n, d) in OPT_META.items()],
            ]),

            # Legend note
            html.Div(className='fade-item', children=[
                html.Div([
                    html.Span('◆ ', style={'color': '#F59E0B', 'fontSize': '10px'}),
                    html.Span(
                        'El marcador final (◆) indica posición de convergencia. '
                        'Mínimo global: (3, 0.5).',
                        style={
                            'fontFamily': "'IBM Plex Sans', sans-serif",
                            'fontSize': '10px', 'color': '#475569', 'lineHeight': '1.6',
                        }
                    ),
                ], style={
                    'background': '#0D1829',
                    'border': '1px solid #F59E0B33',
                    'borderLeft': '3px solid #F59E0B',
                    'borderRadius': '5px',
                    'padding': '10px 12px',
                    'marginTop': '6px',
                }),
            ]),
        ]),
    ]),

    # ── RIGHT PANEL — 3D Plot ──────────────────────────────────────────────
    html.Div(id='right-panel', style={
        'flex': '1',
        'display': 'flex',
        'flexDirection': 'column',
        'background': '#0B1120',
        'overflow': 'hidden',
    }, children=[
        # Top bar
        html.Div(className='top-bar', children=[
            html.Span("Beale Function — Gradient Descent Landscape", style={
                'fontFamily': "'IBM Plex Mono', monospace",
                'fontSize': '10px', 'color': '#1E3050',
                'letterSpacing': '0.1em', 'textTransform': 'uppercase',
            }),
            html.Span("compression: log(J + 1)", style={
                'fontFamily': "'IBM Plex Mono', monospace",
                'fontSize': '10px', 'color': '#1E3050',
            }),
        ], style={
            'display': 'flex', 'justifyContent': 'space-between',
            'alignItems': 'center',
            'padding': '7px 20px',
            'borderBottom': '1px solid #1E3050',
        }),
        dcc.Graph(
            id='plot', figure=INITIAL_FIG,
            style={'flex': '1', 'height': '0', 'minHeight': '100%'},
            config={'displayModeBar': True, 'displaylogo': False,
                    'modeBarButtonsToRemove': ['toImage']},
        ),
    ]),
])


@app.callback(
    Output('plot', 'figure'),
    [Input('lr', 'value'), Input('mu', 'value'),
     Input('beta1', 'value'), Input('beta2', 'value'),
     Input('iters', 'value'),
     Input('opt-sgd', 'value'), Input('opt-momentum', 'value'),
     Input('opt-nesterov', 'value'), Input('opt-adam', 'value')],
    prevent_initial_call=True,
)
def update(lr, mu, b1, b2, n, sgd, mom, nes, adam):
    selected = []
    for key, val in [('sgd', sgd), ('momentum', mom), ('nesterov', nes), ('adam', adam)]:
        if val:
            selected.extend(val)
    return build_figure(lr, mu, b1, b2, n, selected)


if __name__ == '__main__':
    app.run(debug=True)
