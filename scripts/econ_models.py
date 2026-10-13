"""Stage 1, Section 5: econometric benchmarks — AR(1) (mandatory),
BVAR with Minnesota prior (primary comparator), Factor-augmented AR
(secondary). One function per model, all sharing the same signature:
fit on data up to t, forecast h steps ahead for every series.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AutoReg

MAX_HORIZON = 12


def ar1_forecast(y: pd.Series, horizons):
    """AR(1) point forecasts at each horizon, iterated forward."""
    y = y.dropna()
    if len(y) < 10:
        return {h: np.nan for h in horizons}
    model = AutoReg(y.values, lags=1, old_names=False).fit()
    c, phi = model.params
    out, last = {}, y.values[-1]
    val = last
    for h in range(1, max(horizons) + 1):
        val = c + phi * val
        if h in horizons:
            out[h] = val
    return out


def _bvar_fit(df: pd.DataFrame, p=6, lam=0.2):
    """Shared BVAR fit: Minnesota-prior dummy observations + posterior mean
    coefs via lstsq. Returns (B, X, Y, fit_index, columns) so both the
    forecast function and the hybrid model's residual extraction use the
    exact same fit rather than two implementations drifting apart.
    ponytail: closed-form dummy-observation prior (Banbura et al. 2010),
    not a full Gibbs sampler — posterior mean only, which is all Stage 1's
    RMSFE comparison and the hybrid's residual step need.
    """
    data = df.dropna()
    n = data.shape[1]
    if data.shape[0] < p + 10:
        return None

    y = data.values
    sigma = y.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0

    # dummy observations encoding the Minnesota prior: variance shrinks as
    # (lam/lag)^2 (paper Eq. B.1), prior mean 0 since FRED-MD series are
    # already transformed to stationarity (order of integration 0, not 1).
    dummies_x, dummies_y = [], []
    for i in range(n):
        for lag in range(1, p + 1):
            row = np.zeros(n * p + 1)
            row[(lag - 1) * n + i] = lag * sigma[i] / lam
            dummies_x.append(row)
            dummies_y.append(np.zeros(n))
    dx, dy = np.array(dummies_x), np.array(dummies_y)

    def make_lags(mat):
        T = mat.shape[0]
        X = np.ones((T - p, n * p + 1))
        for lag in range(1, p + 1):
            X[:, (lag - 1) * n:lag * n] = mat[p - lag:T - lag]
        X[:, -1] = 1.0
        return X, mat[p:]

    X, Y = make_lags(y)
    X_aug = np.vstack([X, dx])
    Y_aug = np.vstack([Y, dy])
    B = np.linalg.lstsq(X_aug, Y_aug, rcond=None)[0]  # (n*p+1) x n posterior mean coefs
    return B, X, Y, data.index[p:], data.columns


def bvar_minnesota_forecast(df: pd.DataFrame, horizons, p=6, lam=0.2):
    fit = _bvar_fit(df, p, lam)
    if fit is None:
        return {h: {c: np.nan for c in df.columns} for h in horizons}
    B, X, Y, _, columns = fit

    hist = list(Y[-p:])
    out = {}
    for h in range(1, max(horizons) + 1):
        x_new = np.concatenate([np.concatenate(hist[::-1][:p]), [1.0]])
        y_new = x_new @ B
        hist.append(y_new)
        hist = hist[-p:]
        if h in horizons:
            out[h] = dict(zip(columns, y_new))
    return out


def bvar_residuals(df: pd.DataFrame, p=6, lam=0.2) -> pd.DataFrame:
    """In-sample one-step-ahead residuals from the same BVAR fit, for the
    hybrid model's residual-correction step (paper Eq. 12-13)."""
    fit = _bvar_fit(df, p, lam)
    if fit is None:
        return pd.DataFrame(columns=df.columns)
    B, X, Y, idx, columns = fit
    fitted = X @ B
    return pd.DataFrame(Y - fitted, index=idx, columns=columns)


def factor_ar_forecast(df: pd.DataFrame, horizons, n_factors=1, lags=2):
    """PC1 factor + factor-augmented AR, direct multi-horizon (paper's Eq. 6-7):
    y_t = a_h + b_h(L) f_{t-h} + c_h(L) y_{t-h}, one regression per horizon.
    A direct regression (not an iterated one-step model rolled forward) is
    required here: iterating an estimated AR forward h times amplifies any
    |coefficient| slightly over 1 exponentially, which blew up forecasts at
    h=12 in an earlier version of this function.
    ponytail: fixed lag order instead of BIC-selected, per project scope.
    """
    data = df.dropna()
    if data.shape[0] < max(horizons) + lags + 10:
        return {h: {c: np.nan for c in df.columns} for h in horizons}

    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform((data - data.mean()) / data.std())
    fac = factors[:, 0]

    out = {h: {} for h in horizons}
    for col in df.columns:
        y = data[col].values
        n_obs = len(y)
        for h in horizons:
            start = h + lags - 1
            if start >= n_obs:
                continue
            X = np.array([
                [y[tau - h - lag] for lag in range(lags)]
                + [fac[tau - h - lag] for lag in range(lags)]
                + [1.0]
                for tau in range(start, n_obs)
            ])
            target = y[start:n_obs]
            beta = np.linalg.lstsq(X, target, rcond=None)[0]
            x_new = [y[n_obs - 1 - lag] for lag in range(lags)] + [fac[n_obs - 1 - lag] for lag in range(lags)] + [1.0]
            out[h][col] = float(np.dot(beta, x_new))
    return out


def demo():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=200, freq="ME")
    df = pd.DataFrame(rng.normal(size=(200, 3)), index=idx, columns=["a", "b", "c"])

    ar = ar1_forecast(df["a"], [1, 3])
    assert set(ar) == {1, 3} and all(np.isfinite(v) for v in ar.values())

    bv = bvar_minnesota_forecast(df, [1, 3], p=2)
    assert set(bv[1]) == set(df.columns)

    resid = bvar_residuals(df, p=2)
    assert set(resid.columns) == set(df.columns) and len(resid) == len(df) - 2

    fa = factor_ar_forecast(df, [1, 3], lags=2)
    assert set(fa[1]) == set(df.columns)
    print("demo ok")


if __name__ == "__main__":
    demo()
