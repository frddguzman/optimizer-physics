import numpy as np
import torch

# Warm up torch.optim + autograd on first import — eliminates multi-second delay on first callback
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


# Beale function (torch) — global min at (3, 0.5), J = 0
def beale(w):
    x, y = w[0], w[1]
    return ((1.5 - x + x * y) ** 2 +
            (2.25 - x + x * y ** 2) ** 2 +
            (2.625 - x + x * y ** 3) ** 2)


# Beale function (numpy) — for surface mesh rendering
def beale_numpy(X, Y):
    return ((1.5 - X + X * Y) ** 2 +
            (2.25 - X + X * Y ** 2) ** 2 +
            (2.625 - X + X * Y ** 3) ** 2)


def run(opt_type, lr, mu, beta1, beta2, n=200):
    """Run optimizer on Beale's function from fixed start (-4.5, -4.5).

    Returns (xs, ys, js) — trajectory coordinates with log1p-compressed loss.
    """
    w = torch.tensor([-4.5, -4.5], dtype=torch.float32, requires_grad=True)

    # Build optimizer — gate params by type to avoid invalid combinations
    if opt_type == 'sgd':
        opt = torch.optim.SGD([w], lr=lr)
    elif opt_type == 'momentum':
        opt = torch.optim.SGD([w], lr=lr, momentum=mu)
    elif opt_type == 'nesterov':
        # Nesterov requires momentum > 0
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
        loss = beale(w)
        loss.backward()
        # Clip gradients — Beale has ~250k gradient norm at (-4.5,-4.5),
        # which makes SGD/Momentum explode to boundary without clipping
        torch.nn.utils.clip_grad_norm_([w], max_norm=10.0)
        opt.step()
        with torch.no_grad():
            w.clamp_(-5, 5)  # prevent domain escape → NaN
        xs[i] = w[0].item()
        ys[i] = w[1].item()
        # Recompute loss at NEW position so (x, y, z) lies on the surface
        with torch.no_grad():
            js[i] = np.log1p(beale(w).item())

    return xs.tolist(), ys.tolist(), js.tolist()
