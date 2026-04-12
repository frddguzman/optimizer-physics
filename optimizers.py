import numpy as np
import torch

# Warm up torch.optim + autograd — elimina el delay multi-segundo en el primer callback
def _warmup():
    w = torch.tensor([0.0, 0.0], requires_grad=True)
    for Opt in (torch.optim.SGD, torch.optim.Adam):
        o = Opt([w], lr=0.01) if Opt == torch.optim.Adam else Opt([w], lr=0.01, momentum=0.9, nesterov=True)
        for _ in range(3):
            o.zero_grad()
            loss = w[0] ** 2 + w[1] ** 2
            loss.backward()
            o.step()

_warmup()

# ── Namespace seguro para eval con torch ──────────────────────────────────
_TORCH_NS = {k: getattr(torch, k) for k in dir(torch) if not k.startswith('_')}
_TORCH_NS['torch'] = torch


class _TorchNP:
    """Proxy np.* → torch para mantener el grafo de autograd en expresiones custom."""
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


def _eval_torch(expr: str, w):
    """Evalúa expr(x, y) con tensores torch para autograd.
    np.cos/sin/exp/… se redirigen a torch para mantener el grafo."""
    x, y = w[0], w[1]
    ns = {**_TORCH_NS, 'x': x, 'y': y, 'np': _TORCH_NP}
    return eval(compile(expr, '<fn>', 'eval'), {"__builtins__": {}}, ns)  # noqa: S307


# ── Funciones de compatibilidad (la superficie se sigue llamando beale_numpy) ──

def beale(w):
    """Beale function (torch) — global min at (3, 0.5), J = 0."""
    x, y = w[0], w[1]
    return ((1.5 - x + x * y) ** 2 +
            (2.25 - x + x * y ** 2) ** 2 +
            (2.625 - x + x * y ** 3) ** 2)


def beale_numpy(X, Y):
    """Beale function (numpy) — mantenida por compatibilidad."""
    return ((1.5 - X + X * Y) ** 2 +
            (2.25 - X + X * Y ** 2) ** 2 +
            (2.625 - X + X * Y ** 3) ** 2)


def run(opt_type: str, lr: float, mu: float, beta1: float, beta2: float,
        n: int = 200, fn_expr: str | None = None):
    """Ejecuta el optimizador sobre fn_expr (o Beale si fn_expr es None).

    Devuelve (xs, ys, js) — trayectoria con pérdida log1p-comprimida.
    """
    w = torch.tensor([-4.5, -4.5], dtype=torch.float32, requires_grad=True)

    # Selecciona la función de pérdida
    if fn_expr:
        def loss_fn(w_): return _eval_torch(fn_expr, w_)
    else:
        def loss_fn(w_): return beale(w_)

    # Construye el optimizador
    if opt_type == 'sgd':
        opt = torch.optim.SGD([w], lr=lr)
    elif opt_type == 'momentum':
        opt = torch.optim.SGD([w], lr=lr, momentum=mu)
    elif opt_type == 'nesterov':
        opt = torch.optim.SGD([w], lr=lr, momentum=max(mu, 0.01), nesterov=True)
    elif opt_type == 'adam':
        opt = torch.optim.Adam([w], lr=lr, betas=(beta1, beta2))
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    xs = np.empty(n)
    ys = np.empty(n)
    js = np.empty(n)

    for i in range(n):
        opt.zero_grad()
        loss = loss_fn(w)
        loss.backward()
        # Clipping — evita explosión de gradiente en funciones con alta curvatura
        torch.nn.utils.clip_grad_norm_([w], max_norm=10.0)
        opt.step()
        with torch.no_grad():
            w.clamp_(-5, 5)        # evita NaN por escape de dominio
        xs[i] = w[0].item()
        ys[i] = w[1].item()
        with torch.no_grad():
            js[i] = np.log1p(abs(loss_fn(w).item()))

    return xs.tolist(), ys.tolist(), js.tolist()