import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import torch

from optimizers import run

# ── Función activa (mutable en callbacks) ─────────────────────────────────
# Se guarda como string y se compila bajo demanda con eval seguro.

SAFE_NUMPY_NS  = {k: getattr(np,  k) for k in dir(np)  if not k.startswith('_')}
SAFE_TORCH_NS  = {k: getattr(torch, k) for k in dir(torch) if not k.startswith('_')}
SAFE_NUMPY_NS['np'] = np
SAFE_TORCH_NS['torch'] = torch

# ── Funciones preset ───────────────────────────────────────────────────────
PRESETS = {
    'beale': {
        'label': 'Beale  (3, 0.5)',
        'expr':  '(1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2',
    },
    'rosenbrock': {
        'label': 'Rosenbrock  (1, 1)',
        'expr':  '(1 - x)**2 + 100*(y - x**2)**2',
    },
    'himmelblau': {
        'label': "Himmelblau  (3, 2)",
        'expr':  '(x**2 + y - 11)**2 + (x + y**2 - 7)**2',
    },
    'rastrigin': {
        'label': 'Rastrigin  (0, 0)',
        'expr':  '20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))',
    },
    'sphere': {
        'label': 'Esfera  (0, 0)  ← convergencia obvia',
        'expr':  'x**2 + y**2',
    },
    'custom': {
        'label': '✏ Personalizada…',
        'expr':  '',
    },
}
DEFAULT_PRESET = 'beale'

# ── Evaluadores ────────────────────────────────────────────────────────────

def eval_numpy(expr: str, X, Y):
    """Evalúa expr con variables X, Y como arrays numpy."""
    ns = {**SAFE_NUMPY_NS, 'x': X, 'y': Y, 'X': X, 'Y': Y}
    return eval(compile(expr, '<fn>', 'eval'), {"__builtins__": {}}, ns)  # noqa: S307


class _TorchNP:
    """Proxy de np.* que redirige a equivalentes torch para mantener el grafo de autograd.
    Permite escribir np.cos(x) aunque x sea un tensor."""
    cos   = staticmethod(torch.cos)
    sin   = staticmethod(torch.sin)
    tan   = staticmethod(torch.tan)
    exp   = staticmethod(torch.exp)
    log   = staticmethod(torch.log)
    log2  = staticmethod(torch.log2)
    log10 = staticmethod(torch.log10)
    sqrt  = staticmethod(torch.sqrt)
    abs   = staticmethod(torch.abs)
    tanh  = staticmethod(torch.tanh)
    cosh  = staticmethod(torch.cosh)
    sinh  = staticmethod(torch.sinh)
    pi    = 3.141592653589793
    e     = 2.718281828459045

_TORCH_NP = _TorchNP()


def eval_torch(expr: str, w):
    """Evalúa expr con variables x, y como tensores torch (para autograd).
    np.cos, np.sin, etc. se redirigen a torch para mantener el grafo."""
    x, y = w[0], w[1]
    ns = {**SAFE_TORCH_NS, 'x': x, 'y': y, 'np': _TORCH_NP}
    return eval(compile(expr, '<fn>', 'eval'), {"__builtins__": {}}, ns)  # noqa: S307


def validate_expr(expr: str):
    """Devuelve (ok: bool, error_msg: str).  Prueba numpy y torch."""
    if not expr.strip():
        return False, "La expresión está vacía."
    try:
        X_t = np.ones((4, 4), dtype=float)
        result = eval_numpy(expr, X_t, X_t)
        if not np.isfinite(result).any():
            return False, "La función devuelve NaN/Inf en el dominio de prueba."
    except Exception as e:
        return False, f"Error numpy: {e}"
    try:
        w_t = torch.tensor([1.0, 1.0], requires_grad=True)
        loss = eval_torch(expr, w_t)
        loss.backward()
    except Exception as e:
        return False, f"Error torch/autograd: {e}"
    return True, ""

# ── Surface & figure ───────────────────────────────────────────────────────
_lin = np.linspace(-5, 5, 80)
X_GRID, Y_GRID = np.meshgrid(_lin, _lin)

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
        bordercolor='#1E3050', borderwidth=1,
        font=dict(size=12, color='#E2E8F0'),
    ),
)


def build_surface(expr: str):
    Z = np.log1p(np.abs(eval_numpy(expr, X_GRID, Y_GRID)))
    return go.Surface(
        x=_lin, y=_lin, z=Z,
        colorscale='Viridis', opacity=0.7,
        showscale=False, name='f(x,y) log1p',
        hoverinfo='skip',
    )


def build_figure(expr, lr, mu, b1, b2, n, selected):
    traces = [build_surface(expr)]
    for opt in (selected or []):
        xs, ys, js = run(opt, lr, mu, b1, b2, n, fn_expr=expr)
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


INITIAL_EXPR = PRESETS[DEFAULT_PRESET]['expr']
INITIAL_FIG  = build_figure(INITIAL_EXPR, 0.01, 0.9, 0.9, 0.999, 200,
                             ['sgd', 'momentum', 'nesterov', 'adam'])

# ── CSS ────────────────────────────────────────────────────────────────────
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

#left-panel  { animation: panelSlideIn 0.5s cubic-bezier(0.22,1,0.36,1) both; }
#right-panel { animation: plotFadeIn  0.7s cubic-bezier(0.22,1,0.36,1) 0.15s both; }

.header-icon { animation: headerGlow 3s ease-in-out infinite; display: inline-block; }

.fade-item { opacity: 0; animation: fadeUp 0.45s ease forwards; }
.fade-item:nth-child(1) { animation-delay: 0.10s; }
.fade-item:nth-child(2) { animation-delay: 0.18s; }
.fade-item:nth-child(3) { animation-delay: 0.26s; }
.fade-item:nth-child(4) { animation-delay: 0.34s; }
.fade-item:nth-child(5) { animation-delay: 0.42s; }
.fade-item:nth-child(6) { animation-delay: 0.50s; }
.fade-item:nth-child(7) { animation-delay: 0.58s; }
.fade-item:nth-child(8) { animation-delay: 0.66s; }

.opt-card { transition: transform 0.18s ease, box-shadow 0.18s ease; }
.opt-card:hover { transform: translateX(3px); box-shadow: -3px 0 12px rgba(0,0,0,0.4); }

.opt-dot { animation: dotPulse 2.5s ease-in-out infinite; }
.opt-dot-sgd      { color: #EF4444; animation-delay: 0.0s; }
.opt-dot-momentum { color: #3B82F6; animation-delay: 0.6s; }
.opt-dot-nesterov { color: #10B981; animation-delay: 1.2s; }
.opt-dot-adam     { color: #F59E0B; animation-delay: 1.8s; }

.top-bar { position: relative; overflow: hidden; }
.top-bar::after {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(transparent 0%, rgba(14,165,233,0.03) 50%, transparent 100%);
  background-size: 100% 40px;
  animation: scanline 4s linear infinite;
  pointer-events: none;
}

/* ── Slider ── */
.rc-slider-rail  { background: var(--border) !important; height: 3px !important; }
.rc-slider-track { background: var(--teal)   !important; height: 3px !important; transition: width 0.12s ease !important; }
.rc-slider-handle {
  border-color: var(--teal) !important; background: var(--navy) !important;
  box-shadow: 0 0 0 2px var(--teal) !important;
  width: 14px !important; height: 14px !important; margin-top: -5px !important;
  transition: box-shadow 0.15s ease !important;
}
.rc-slider-handle:hover { box-shadow: 0 0 0 5px rgba(14,165,233,0.22) !important; }
.rc-slider-tooltip-inner {
  background: #131D30 !important; color: #0EA5E9 !important;
  font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important;
  border: 1px solid #1E3050 !important; border-radius: 4px !important;
  padding: 2px 8px !important; box-shadow: none !important;
}

/* ── Checkbox ── */
input[type="checkbox"] {
  appearance: none; -webkit-appearance: none;
  width: 15px; height: 15px; border: 1.5px solid #1E3050; border-radius: 3px;
  background: #0B1120; cursor: pointer; flex-shrink: 0;
  transition: background 0.15s, border-color 0.15s, box-shadow 0.15s; position: relative;
}
input[type="checkbox"]:hover { border-color: var(--teal); box-shadow: 0 0 0 3px rgba(14,165,233,0.15); }
input[type="checkbox"]:checked { background: #0EA5E9; border-color: #0EA5E9; }
input[type="checkbox"]:checked::after {
  content: ''; position: absolute; left: 3px; top: 1px;
  width: 5px; height: 8px; border: 2px solid white;
  border-top: none; border-left: none; transform: rotate(45deg);
}

/* ── Preset dropdown (dcc.Dropdown) ── */
.Select-control, .Select-menu-outer {
  background: #0D1829 !important; border-color: #1E3050 !important;
  color: #94A3B8 !important; border-radius: 5px !important;
}
.Select-value-label, .Select-placeholder { color: #94A3B8 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; }
.Select-option { background: #0D1829 !important; color: #94A3B8 !important; font-size: 11px !important; }
.Select-option:hover, .Select-option.is-focused { background: #131D30 !important; color: #0EA5E9 !important; }
.Select-arrow { border-top-color: #334155 !important; }

/* ── Custom function textarea ── */
#fn-input {
  width: 100%; background: #0B1120; color: #E2E8F0;
  border: 1px solid #1E3050; border-radius: 5px;
  font-family: 'IBM Plex Mono', monospace; font-size: 12px;
  padding: 8px 10px; resize: vertical; min-height: 64px;
  transition: border-color 0.2s, box-shadow 0.2s; outline: none;
  line-height: 1.5;
}
#fn-input:focus { border-color: #0EA5E9; box-shadow: 0 0 0 3px rgba(14,165,233,0.12); }
#fn-input.fn-error { border-color: #EF4444 !important; box-shadow: 0 0 0 3px rgba(239,68,68,0.12) !important; }
#fn-input.fn-ok    { border-color: #10B981 !important; box-shadow: 0 0 0 3px rgba(16,185,129,0.12) !important; }

/* ── Apply button ── */
#fn-apply {
  width: 100%; margin-top: 8px; padding: 7px 0;
  background: #0EA5E9; color: white; border: none; border-radius: 5px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
  letter-spacing: 0.08em; cursor: pointer;
  transition: background 0.15s, transform 0.1s, box-shadow 0.15s;
}
#fn-apply:hover { background: #38BDF8; box-shadow: 0 0 12px rgba(14,165,233,0.35); }
#fn-apply:active { transform: scale(0.97); }
"""

# ── Layout helpers ─────────────────────────────────────────────────────────

def card(children, accent=None, extra_style=None):
    base = {
        'background': '#131D30',
        'border': '1px solid ' + (accent + '44' if accent else '#1E3050'),
        'borderLeft': '3px solid ' + (accent if accent else '#1E3050'),
        'borderRadius': '6px', 'padding': '14px 16px', 'marginBottom': '12px',
    }
    if extra_style:
        base.update(extra_style)
    return html.Div(children, style=base)


def label_text(text):
    return html.P(text, style={
        'fontFamily': "'IBM Plex Sans', sans-serif", 'fontSize': '10px',
        'fontWeight': '600', 'color': '#94A3B8', 'letterSpacing': '0.1em',
        'textTransform': 'uppercase', 'marginBottom': '8px',
    })


def info_text(text, color='#475569'):
    return html.P(text, style={
        'fontFamily': "'IBM Plex Sans', sans-serif", 'fontSize': '11px',
        'color': color, 'lineHeight': '1.6', 'marginTop': '6px',
    })


def section_header(icon, title):
    return html.Div([
        html.Span(icon, style={'fontSize': '12px', 'marginRight': '7px', 'color': '#0EA5E9'}),
        html.Span(title, style={
            'fontFamily': "'IBM Plex Mono', monospace", 'fontSize': '10px',
            'fontWeight': '600', 'letterSpacing': '0.14em',
            'textTransform': 'uppercase', 'color': '#0EA5E9',
        }),
    ], style={'marginBottom': '10px', 'marginTop': '8px'})


def slider_block(slider_id, label, description, **kwargs):
    return html.Div([
        label_text(label),
        dcc.Slider(id=slider_id, marks=None,
                   tooltip={"placement": "bottom", "always_visible": False}, **kwargs),
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
                'background': color, 'flexShrink': '0', 'color': color, 'marginTop': '2px',
            }),
            html.Div([
                html.Span(name, style={
                    'fontFamily': "'IBM Plex Mono', monospace", 'fontSize': '12px',
                    'fontWeight': '600', 'color': color, 'display': 'block',
                }),
                html.Span(desc, style={
                    'fontFamily': "'IBM Plex Sans', sans-serif", 'fontSize': '10px',
                    'color': '#475569', 'display': 'block', 'marginTop': '1px',
                }),
            ], style={'flex': '1'}),
            dcc.Checklist(
                id=f'opt-{key}', options=[{'label': '', 'value': key}], value=[key],
                style={'display': 'flex', 'alignItems': 'center'},
                inputStyle={'cursor': 'pointer'},
            ),
        ], style={'display': 'flex', 'alignItems': 'flex-start', 'gap': '10px'}),
    ], className='opt-card', style={
        'background': '#0D1829', 'border': f'1px solid {color}22',
        'borderLeft': f'3px solid {color}', 'borderRadius': '5px',
        'padding': '10px 12px', 'marginBottom': '7px',
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
    'display': 'flex', 'background': '#0B1120', 'height': '100vh',
    'fontFamily': "'IBM Plex Sans', sans-serif", 'overflow': 'hidden',
}, children=[

    # ── Store: expresión activa ──────────────────────────────────────────
    dcc.Store(id='active-expr', data=INITIAL_EXPR),

    # ── LEFT PANEL ──────────────────────────────────────────────────────
    html.Div(id='left-panel', style={
        'width': '290px', 'minWidth': '290px', 'background': '#111827',
        'borderRight': '1px solid #1E3050', 'display': 'flex',
        'flexDirection': 'column', 'overflowY': 'auto', 'overflowX': 'hidden',
    }, children=[

        # Header
        html.Div([
            html.Div([
                html.Span("◈ ", className='header-icon', style={'color': '#0EA5E9', 'fontSize': '16px'}),
                html.Span("Optimizer Physics", style={
                    'fontFamily': "'IBM Plex Mono', monospace",
                    'fontSize': '14px', 'fontWeight': '600', 'color': '#F1F5F9',
                }),
            ], style={'marginBottom': '8px'}),
            info_text("Visualización 3D del descenso de distintos optimizadores "
                      "sobre una función de pérdida configurable.", color='#334155'),
        ], style={
            'padding': '18px 18px 14px', 'borderBottom': '1px solid #1E3050',
            'background': 'linear-gradient(180deg, #131D30 0%, #111827 100%)',
        }),

        # Body
        html.Div(style={'padding': '14px 16px', 'flex': '1'}, children=[

            # ── FUNCIÓN ─────────────────────────────────────────────────
            html.Div(className='fade-item', children=[
                section_header('ƒ', 'Función de pérdida'),
                card([
                    label_text('Preset'),
                    dcc.Dropdown(
                        id='fn-preset',
                        options=[{'label': v['label'], 'value': k} for k, v in PRESETS.items()],
                        value=DEFAULT_PRESET,
                        clearable=False,
                        style={'marginBottom': '10px'},
                    ),
                    label_text('Expresión  f(x, y)'),
                    dcc.Textarea(
                        id='fn-input',
                        value=INITIAL_EXPR,
                        placeholder='(1.5 - x + x*y)**2 + ...',
                        style={
                            'width': '100%', 'background': '#0B1120', 'color': '#E2E8F0',
                            'border': '1px solid #1E3050', 'borderRadius': '5px',
                            'fontFamily': "'IBM Plex Mono', monospace", 'fontSize': '12px',
                            'padding': '8px 10px', 'resize': 'vertical',
                            'minHeight': '64px', 'outline': 'none', 'lineHeight': '1.5',
                        },
                    ),
                    html.Button('▶  Aplicar función', id='fn-apply', n_clicks=0, style={
                        'width': '100%', 'marginTop': '8px', 'padding': '7px 0',
                        'background': '#0EA5E9', 'color': 'white', 'border': 'none',
                        'borderRadius': '5px', 'fontFamily': "'IBM Plex Mono', monospace",
                        'fontSize': '11px', 'fontWeight': '600', 'letterSpacing': '0.08em',
                        'cursor': 'pointer',
                    }),
                    # Error / status banner
                    html.Div(id='fn-status', style={'marginTop': '8px'}),
                    info_text('Usa x, y como variables. Disponible: np.sin, np.cos, np.exp, **…',
                              color='#334155'),
                ], accent='#8B5CF6'),
            ]),

            # ── HIPERPARÁMETROS ──────────────────────────────────────────
            html.Div(className='fade-item', children=[section_header('⚙', 'Hiperparámetros')]),

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

            # ── OPTIMIZADORES ────────────────────────────────────────────
            html.Div(className='fade-item', children=[section_header('◉', 'Optimizadores activos')]),

            html.Div(className='fade-item', children=[
                *[optimizer_toggle(k, c, n, d) for k, (c, n, d) in OPT_META.items()],
            ]),

            html.Div(className='fade-item', children=[
                html.Div([
                    html.Span('◆ ', style={'color': '#F59E0B', 'fontSize': '10px'}),
                    html.Span('El marcador final (◆) indica posición de convergencia.',
                              style={'fontFamily': "'IBM Plex Sans', sans-serif",
                                     'fontSize': '10px', 'color': '#475569', 'lineHeight': '1.6'}),
                ], style={
                    'background': '#0D1829', 'border': '1px solid #F59E0B33',
                    'borderLeft': '3px solid #F59E0B', 'borderRadius': '5px',
                    'padding': '10px 12px', 'marginTop': '6px',
                }),
            ]),
        ]),
    ]),

    # ── RIGHT PANEL ─────────────────────────────────────────────────────
    html.Div(id='right-panel', style={
        'flex': '1', 'display': 'flex', 'flexDirection': 'column',
        'background': '#0B1120', 'overflow': 'hidden',
    }, children=[
        html.Div(className='top-bar', children=[
            html.Span(id='top-fn-label', children="f(x,y) — Beale Function", style={
                'fontFamily': "'IBM Plex Mono', monospace", 'fontSize': '10px',
                'color': '#1E3050', 'letterSpacing': '0.1em', 'textTransform': 'uppercase',
            }),
            html.Span("compression: log(J + 1)", style={
                'fontFamily': "'IBM Plex Mono', monospace",
                'fontSize': '10px', 'color': '#1E3050',
            }),
        ], style={
            'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
            'padding': '7px 20px', 'borderBottom': '1px solid #1E3050',
        }),
        dcc.Graph(
            id='plot', figure=INITIAL_FIG,
            style={'flex': '1', 'height': '0', 'minHeight': '100%'},
            config={'displayModeBar': True, 'displaylogo': False,
                    'modeBarButtonsToRemove': ['toImage']},
        ),
    ]),
])


# ── Callback 1: preset → rellena textarea ─────────────────────────────────
@app.callback(
    Output('fn-input', 'value'),
    Input('fn-preset', 'value'),
    prevent_initial_call=True,
)
def fill_from_preset(preset):
    if preset == 'custom':
        return ''
    return PRESETS[preset]['expr']


# ── Callback 2: Aplicar función (botón) → valida, guarda en Store ─────────
@app.callback(
    Output('active-expr', 'data'),
    Output('fn-status', 'children'),
    Output('top-fn-label', 'children'),
    Input('fn-apply', 'n_clicks'),
    State('fn-input', 'value'),
    State('fn-preset', 'value'),
    prevent_initial_call=True,
)
def apply_function(n_clicks, expr, preset):
    if not expr or not expr.strip():
        return (
            INITIAL_EXPR,
            html.Div("⚠ Expresión vacía.", style={
                'color': '#EF4444', 'fontSize': '11px',
                'fontFamily': "'IBM Plex Mono', monospace",
            }),
            "f(x,y) — expresión vacía",
        )

    ok, err = validate_expr(expr)
    if not ok:
        return (
            INITIAL_EXPR,
            html.Div([
                html.Span("✕ Error: ", style={'fontWeight': '600'}),
                html.Span(err),
            ], style={
                'color': '#EF4444', 'fontSize': '10px',
                'fontFamily': "'IBM Plex Mono', monospace",
                'background': '#1a0a0a', 'border': '1px solid #EF444433',
                'borderLeft': '3px solid #EF4444', 'borderRadius': '4px',
                'padding': '6px 10px', 'lineHeight': '1.5',
            }),
            "f(x,y) — error en expresión",
        )

    # Éxito
    preset_label = PRESETS.get(preset, {}).get('label', 'Custom')
    top_label = f"f(x,y) — {preset_label}" if preset != 'custom' else f"f(x,y) — {expr[:40]}…"
    status = html.Div("✓ Función aplicada correctamente.", style={
        'color': '#10B981', 'fontSize': '11px',
        'fontFamily': "'IBM Plex Mono', monospace",
    })
    return expr.strip(), status, top_label


# ── Callback 3: actualizar gráfico ────────────────────────────────────────
@app.callback(
    Output('plot', 'figure'),
    [Input('active-expr', 'data'),
     Input('lr', 'value'), Input('mu', 'value'),
     Input('beta1', 'value'), Input('beta2', 'value'),
     Input('iters', 'value'),
     Input('opt-sgd', 'value'), Input('opt-momentum', 'value'),
     Input('opt-nesterov', 'value'), Input('opt-adam', 'value')],
    prevent_initial_call=True,
)
def update(expr, lr, mu, b1, b2, n, sgd, mom, nes, adam):
    selected = []
    for key, val in [('sgd', sgd), ('momentum', mom), ('nesterov', nes), ('adam', adam)]:
        if val:
            selected.extend(val)
    try:
        return build_figure(expr or INITIAL_EXPR, lr, mu, b1, b2, n, selected)
    except Exception:
        return INITIAL_FIG


if __name__ == '__main__':
    app.run(debug=True)