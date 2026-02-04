import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Metrics
# ----------------------------
def centroid_norm(X):
    mu = X.mean(axis=0)
    return np.linalg.norm(mu), mu

def cosine_percentiles_to_centroid(X, ps=(10, 50, 90)):
    cn, mu = centroid_norm(X)
    if cn < 1e-12:
        return {f"cos_p{p}": np.nan for p in ps}, cn
    c_hat = mu / cn
    cos = X @ c_hat
    return {f"cos_p{p}": float(np.percentile(cos, p)) for p in ps}, cn

def effective_rank(X):
    # effective rank of covariance of centered data
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, (len(Xc) - 1))
    evals = np.linalg.eigvalsh(C)
    evals = np.clip(evals, 0.0, None)
    s = evals.sum()
    if s < 1e-12:
        return 1.0
    p = evals / s
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))

# ----------------------------
# Sampling helpers on S^2 (3D)
# ----------------------------
def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)

def _orthonormal_basis(u):
    u = _unit(u)
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(u, a))
    e2 = np.cross(u, e1)
    return u, e1, e2

def _sample_cap(rng, center, n, cap_deg):
    center, e1, e2 = _orthonormal_basis(center)
    alpha = np.deg2rad(cap_deg)
    z = rng.uniform(np.cos(alpha), 1.0, size=n)  # cos(theta) uniform
    theta = np.arccos(z)
    phi = rng.uniform(0, 2*np.pi, size=n)
    X = (np.cos(theta)[:, None] * center[None, :]
         + np.sin(theta)[:, None] * (np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :]))
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def _fibonacci_sphere_dirs(k):
    if k == 1:
        return np.array([[0.0, 0.0, 1.0]])
    i = np.arange(k)
    phi = (1 + 5**0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2*(i + 0.5)/k
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)

def sample_line(rng, n=1200, sigma=0.02):
    # 1D-ish around one direction
    u = _unit(rng.normal(size=3))
    X = u + sigma * rng.normal(size=(n, 3))
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def sample_cap(rng, n=1200, cap_deg=25):
    u = _unit(rng.normal(size=3))
    return _sample_cap(rng, u, n, cap_deg)

def sample_mixture(rng, n=1200, k=4, spread_deg=10, balance=1.0):
    # balance in (0,1]: 1.0 = equal; smaller => one mode dominates
    dirs = _fibonacci_sphere_dirs(k)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    dirs = (Q @ dirs.T).T

    # weights: one dominant if balance small
    w = np.ones(k)
    w[0] = 1.0 / max(1e-9, balance)  # make mode 0 heavier as balance decreases
    w = w / w.sum()
    counts = rng.multinomial(n, w)

    Xs = []
    for i in range(k):
        if counts[i] == 0:
            continue
        Xs.append(_sample_cap(rng, dirs[i], counts[i], spread_deg))
    X = np.vstack(Xs)
    return X

# ----------------------------
# Scenario sweep + plotting
# ----------------------------
def compute_metrics(X):
    cn, _ = centroid_norm(X)
    er = effective_rank(X)
    cosp, _ = cosine_percentiles_to_centroid(X)
    return cn, er, cosp["cos_p10"]

def _plot_sphere(ax, n=18, alpha=0.10):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.4, alpha=alpha)

def plot_cloud(ax, X, title):
    cn, mu = centroid_norm(X)
    er = effective_rank(X)
    cosp, _ = cosine_percentiles_to_centroid(X)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=6, alpha=0.35)
    ax.quiver(0, 0, 0, mu[0], mu[1], mu[2], linewidth=2)
    _plot_sphere(ax)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"{title}\ncn={cn:.2f} | er={er:.2f} | cos_p10={cosp['cos_p10']:.2f}", pad=10)

def sweep_and_visualize(seed=0, n=1200):
    rng = np.random.default_rng(seed)

    records = []  # (cn, er, cos_p10, meta, X_small)
    # Sweep knobs
    sigmas = [0.005, 0.01, 0.02, 0.05, 0.10]                 # line noise
    cap_degs = [5, 10, 20, 35, 60, 90, 120, 160]             # cap size
    ks = [2, 3, 4, 6, 8]                                      # mixture modes
    spread_degs = [3, 6, 10, 20, 35]                          # within-mode spread
    balances = [1.0, 0.5, 0.25, 0.1, 0.05]                    # mixture dominance

    # Line regime
    for s in sigmas:
        X = sample_line(rng, n=n, sigma=s)
        cn, er, p10 = compute_metrics(X)
        records.append((cn, er, p10, ("line", {"sigma": s}), X[:: max(1, n//600)]))

    # Cap regime
    for a in cap_degs:
        X = sample_cap(rng, n=n, cap_deg=a)
        cn, er, p10 = compute_metrics(X)
        records.append((cn, er, p10, ("cap", {"cap_deg": a}), X[:: max(1, n//600)]))

    # Mixture regime
    for k in ks:
        for sp in spread_degs:
            for bal in balances:
                X = sample_mixture(rng, n=n, k=k, spread_deg=sp, balance=bal)
                cn, er, p10 = compute_metrics(X)
                records.append((cn, er, p10, ("mixture", {"k": k, "spread_deg": sp, "balance": bal}), X[:: max(1, n//600)]))

    # Convert to arrays for plotting
    cn = np.array([r[0] for r in records])
    er = np.array([r[1] for r in records])
    p10 = np.array([r[2] for r in records])

    # --- Figure 1: metric landscape
    fig1 = plt.figure(figsize=(10, 7))
    ax = fig1.add_subplot(1, 1, 1)
    sc = ax.scatter(cn, er, c=p10, s=35, alpha=0.85)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("cos_p10 (to centroid direction)")

    ax.set_xlabel("centroid_norm")
    ax.set_ylabel("effective rank")
    ax.set_title("Scenario landscape: centroid_norm vs effective-rank (color = cos_p10)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    # --- Pick 6 representative scenarios (corners + middle)
    # Targets: (high cn, low er), (high cn, high er), (low cn, high er), (low cn, low er), plus 2 middles
    targets = [
        ("High cn / Low er",  np.quantile(cn, 0.90), np.quantile(er, 0.10)),
        ("High cn / High er", np.quantile(cn, 0.90), np.quantile(er, 0.90)),
        ("Low cn / High er",  np.quantile(cn, 0.10), np.quantile(er, 0.90)),
        ("Low cn / Low er",   np.quantile(cn, 0.10), np.quantile(er, 0.10)),
        ("Middle-ish #1",     np.quantile(cn, 0.50), np.quantile(er, 0.50)),
        ("Middle-ish #2",     np.quantile(cn, 0.65), np.quantile(er, 0.35)),
    ]

    chosen = []
    used = set()
    for name, tcn, ter in targets:
        dist = (cn - tcn)**2 + (er - ter)**2
        idx = int(np.argmin(dist))
        # avoid duplicates
        if idx in used:
            order = np.argsort(dist)
            for j in order:
                if int(j) not in used:
                    idx = int(j)
                    break
        used.add(idx)
        chosen.append((name, records[idx]))

    # --- Figure 2: representative point clouds on sphere
    fig2 = plt.figure(figsize=(14, 9))
    for i, (name, rec) in enumerate(chosen):
        _, _, _, meta, Xsmall = rec
        reg, params = meta
        ax3d = fig2.add_subplot(2, 3, i + 1, projection="3d")
        title = f"{name}\n{reg} {params}"
        plot_cloud(ax3d, Xsmall, title)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sweep_and_visualize(seed=0, n=1500)
