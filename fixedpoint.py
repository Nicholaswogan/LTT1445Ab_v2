from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    iters: int
    func_evals: int
    # (k, x_k, r_k, ||r_k||_scaled, omega_k, beta_k)
    history: List[Tuple[int, np.ndarray, np.ndarray, float, float, float]]


class RobustFixedPointSolver:
    """
    Robust Anderson-accelerated fixed-point solver.

    This class solves

    .. math::
        x = g(x)

    by iterating on the residual

    .. math::
        r(x) = g(x) - x.

    Notes
    -----
    Per iteration, the algorithm:
    1. Evaluates ``g(x)`` once.
    2. Forms a relaxed fixed-point proposal ``x_plain = x + omega * r``.
    3. Builds an Anderson candidate from recent ``(x, r)`` history.
    4. Mixes using ``beta`` and applies a directional safeguard.
    5. Applies optional step limits and updates adaptive ``omega``/``beta``.
    """

    def __init__(
        self,
        g: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        *,
        m: int = 6,
        omega: float = 0.5,
        omega_min: float = 0.05,
        omega_max: float = 1.0,
        omega_shrink: float = 0.5,
        omega_grow: float = 1.2,
        beta: float = 1.0,
        beta_min: float = 0.1,
        beta_shrink: float = 0.5,
        beta_grow: float = 1.1,
        ridge: float = 1e-6,
        max_step: float | None = None,
        max_norm_step: float | None = None,
        growth_threshold: float = 2.0,
        improve_threshold: float = 0.5,
        safeguard_factor: float = 1.2,
        scale: float | Sequence[float] = 1.0,
        tol: float = 1e-8,
        max_tol: float | None = None,
        max_iter: int = 80,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the robust Anderson fixed-point solver.

        Parameters
        ----------
        g : callable
            Fixed-point map. Must take and return a 1D NumPy array of the same
            shape.
        x0 : ndarray
            Initial guess as a 1D NumPy array. Use a length-1 array for scalar
            problems.
        m : int, optional
            Anderson memory length.
        omega : float, optional
            Initial under-relaxation for the plain fixed-point step.
        omega_min, omega_max : float, optional
            Bounds for adaptive ``omega``.
        omega_shrink, omega_grow : float, optional
            Multipliers used when residual worsens/improves.
        beta : float, optional
            Initial blend between plain and Anderson proposals.
        beta_min : float, optional
            Lower bound for adaptive ``beta``.
        beta_shrink, beta_grow : float, optional
            Multipliers used when residual worsens/improves.
        ridge : float, optional
            Ridge regularization in the Anderson least-squares system.
        max_step : float or None, optional
            Optional per-component cap on ``x_{k+1}-x_k``.
        max_norm_step : float or None, optional
            Optional cap on scaled RMS step norm.
        growth_threshold : float, optional
            If residual grows above this ratio, shrink damping and restart
            Anderson history.
        improve_threshold : float, optional
            If residual shrinks below this ratio, cautiously grow damping.
        safeguard_factor : float, optional
            Directional safeguard aggressiveness; values near 1 are more
            conservative.
        scale : float or array_like, optional
            Scaling used in residual/step RMS norms.
        tol : float, optional
            Convergence tolerance on scaled RMS residual norm.
        max_tol : float or None, optional
            Optional max per-component scaled residual tolerance.
        max_iter : int, optional
            Maximum number of iterations.
        verbose : bool, optional
            If True, prints per-iteration diagnostics.
        """
        if m < 0:
            raise ValueError("m must be >= 0")
        if not (0.0 < omega_min <= omega <= omega_max <= 1.0):
            raise ValueError("require 0 < omega_min <= omega <= omega_max <= 1")
        if beta <= 0.0:
            raise ValueError("beta must be > 0")
        if beta_min <= 0.0:
            raise ValueError("beta_min must be > 0")
        if ridge < 0.0:
            raise ValueError("ridge must be >= 0")
        if max_step is not None and max_step <= 0.0:
            raise ValueError("max_step must be > 0 or None")
        if max_norm_step is not None and max_norm_step <= 0.0:
            raise ValueError("max_norm_step must be > 0 or None")
        if safeguard_factor < 1.0:
            raise ValueError("safeguard_factor must be >= 1.0")
        if max_tol is not None and max_tol <= 0.0:
            raise ValueError("max_tol must be > 0 or None")
        if not isinstance(x0, np.ndarray):
            raise TypeError("x0 must be a numpy.ndarray (use a length-1 array for scalars)")

        x = np.asarray(x0, dtype=float)
        if x.ndim != 1:
            raise ValueError("x0 must be a 1D numpy array")
        sc = (
            np.full_like(x, float(scale), dtype=float)
            if isinstance(scale, (int, float))
            else np.asarray(scale, dtype=float)
        )
        if sc.shape != x.shape:
            raise ValueError("scale must be a scalar or have the same shape as x0")
        if np.any(sc <= 0.0):
            raise ValueError("scale entries must be > 0")

        self.g = g
        self.x = x
        self.sc = sc
        self.m = int(m)
        self.omega = float(omega)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.omega_shrink = float(omega_shrink)
        self.omega_grow = float(omega_grow)
        self.beta = float(beta)
        self.beta_min = float(beta_min)
        self.beta_shrink = float(beta_shrink)
        self.beta_grow = float(beta_grow)
        self.ridge = float(ridge)
        self.max_step = max_step
        self.max_norm_step = max_norm_step
        self.growth_threshold = float(growth_threshold)
        self.improve_threshold = float(improve_threshold)
        self.safeguard_factor = float(safeguard_factor)
        self.tol = float(tol)
        self.max_tol = max_tol
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)

        self.history: List[Tuple[int, np.ndarray, np.ndarray, float, float, float]] = []
        self.func_evals = 0
        self.xs: List[np.ndarray] = []
        self.rs: List[np.ndarray] = []
        self.prev_norm: float | None = None
        self.prev_x: np.ndarray | None = None
        self.prev_r: np.ndarray | None = None
        self.k = 0
        self.converged = False
        self.terminated = False

    def _rms_scaled(self, v: np.ndarray) -> float:
        vs = v / self.sc
        return float(np.linalg.norm(vs) / (max(1, vs.size) ** 0.5))

    def _max_scaled(self, v: np.ndarray) -> float:
        return float(np.max(np.abs(v / self.sc)))

    def step(self) -> bool:
        """
        Execute one solver iteration.

        Returns
        -------
        bool
            True if the solver is converged after this call, else False.
        """
        if self.converged or self.k >= self.max_iter:
            return self.converged

        k = self.k
        # One expensive model call: evaluate the fixed-point map at the current state.
        gx = np.asarray(self.g(self.x), dtype=float)
        if gx.shape != self.x.shape:
            raise ValueError("g(x) must return a numpy array with the same shape as x0")
        self.func_evals += 1
        if not np.all(np.isfinite(gx)):
            if self.verbose:
                print(f"[AA] k={k:3d}  non-finite g(x) detected; stopping (non-converged)")
            self.terminated = True
            return False

        # Residual for the root-equivalent problem r(x) = g(x) - x.
        xk = self.x.copy()
        r = gx - self.x
        rnorm = self._rms_scaled(r)
        rmax = self._max_scaled(r)

        # Convergence test: scaled RMS criterion plus optional per-component max criterion.
        if rnorm < self.tol and (self.max_tol is None or rmax < self.max_tol):
            self.history.append((k, xk, r.copy(), rnorm, self.omega, self.beta))
            if self.verbose:
                print(
                    f"[AA] k={k:3d}  rnorm={rnorm: .3e}  rmax={rmax: .3e}  "
                    f"omega={self.omega: .3f}  beta={self.beta: .3f}  (converged)"
                )
            self.converged = True
            return True

        # Adapt relaxation/mixing using residual progress relative to last iteration.
        # If things get worse, damp and restart AA history; if much better, grow cautiously.
        did_restart = False
        if self.prev_norm is not None and self.prev_norm > 0.0:
            ratio = rnorm / self.prev_norm
            if ratio > self.growth_threshold:
                self.omega = max(self.omega_min, self.omega * self.omega_shrink)
                self.beta = max(self.beta_min, self.beta * self.beta_shrink)
                self.xs.clear()
                self.rs.clear()
                self.prev_x = None
                self.prev_r = None
                did_restart = True
            elif ratio < self.improve_threshold:
                self.omega = min(self.omega_max, self.omega * self.omega_grow)
                self.beta = min(1.0, self.beta * self.beta_grow)
        self.prev_norm = rnorm

        # Append current point to limited-memory buffers used by Anderson mixing.
        self.xs.append(xk)
        self.rs.append(r.copy())

        # Baseline robust step: under-relaxed fixed-point update.
        omega_used = self.omega
        x_plain = self.x + self.omega * r

        # Accelerated step: Anderson type-I on recent residual/iterate differences.
        # Falls back to plain g(x) if there is not enough history.
        x_acc = gx.copy()
        mk_used = 0
        if self.m > 0 and len(self.xs) >= 2:
            mk = min(self.m, len(self.xs) - 1)
            mk_used = mk
            i0 = len(self.xs) - (mk + 1)
            x_win = self.xs[i0:]
            r_win = self.rs[i0:]

            dR = np.column_stack([r_win[j + 1] - r_win[j] for j in range(mk)])
            dX = np.column_stack([x_win[j + 1] - x_win[j] for j in range(mk)])
            dR_scaled = dR / self.sc[:, None]
            r_scaled = r / self.sc
            A = dR_scaled.T @ dR_scaled + self.ridge * np.eye(mk)
            b = dR_scaled.T @ r_scaled
            try:
                gamma = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                gamma = np.linalg.lstsq(A, b, rcond=None)[0]
            x_acc = gx - dX @ gamma

        # Blend baseline and accelerated candidates.
        x_next = x_plain + self.beta * (x_acc - x_plain)

        # Directional safeguard (no extra g-eval):
        # use a local secant slope estimate to reject overly aggressive AA directions.
        did_safeguard = False
        beta_before_safeguard = self.beta
        if self.prev_x is not None and self.prev_r is not None:
            last_dx = self.x - self.prev_x
            last_dr = r - self.prev_r
            dxs = last_dx / self.sc
            drs = last_dr / self.sc
            denom = float(dxs @ dxs)
            if denom > 0.0:
                alpha = float(drs @ dxs) / denom
                base = self._rms_scaled(r + alpha * (x_plain - self.x))
                beta_try = self.beta
                while beta_try >= self.beta_min:
                    x_try = x_plain + beta_try * (x_acc - x_plain)
                    pred = self._rms_scaled(r + alpha * (x_try - self.x))
                    if pred <= self.safeguard_factor * base:
                        x_next = x_try
                        self.beta = beta_try
                        break
                    beta_try *= self.beta_shrink
                else:
                    self.beta = self.beta_min
                    x_next = x_plain
                    self.xs.clear()
                    self.rs.clear()
                did_safeguard = self.beta != beta_before_safeguard

        # Optional hard step limits (componentwise and/or scaled RMS norm).
        beta_used = self.beta
        dx = x_next - self.x
        dx_before_clip = dx.copy()
        if self.max_step is not None:
            dx = np.clip(dx, -float(self.max_step), float(self.max_step))
        if self.max_norm_step is not None:
            nrm = self._rms_scaled(dx)
            cap = float(self.max_norm_step)
            if nrm > cap and nrm > 0.0:
                dx *= cap / nrm
        did_clip = not np.allclose(dx, dx_before_clip)

        # Commit the step and keep state needed for next-iteration safeguards.
        x_new = self.x + dx
        self.prev_x = self.x
        self.prev_r = r
        self.x = x_new

        # Store iteration diagnostics/history for analysis and plotting.
        self.history.append((k, xk, r.copy(), rnorm, omega_used, beta_used))
        dxnorm = self._rms_scaled(dx)
        if self.verbose:
            flags = []
            if mk_used == 0:
                flags.append("noAA")
            if did_restart:
                flags.append("restart")
            if did_safeguard:
                flags.append("safeguard")
            if did_clip:
                flags.append("clip")
            flag_str = ("  [" + ",".join(flags) + "]") if flags else ""
            print(
                f"[AA] k={k:3d}  rnorm={rnorm: .3e}  rmax={rmax: .3e}  dxnorm={dxnorm: .3e}  "
                f"omega={omega_used: .3f}  beta={beta_used: .3f}  mk={mk_used}{flag_str}"
            )

        self.k += 1
        return False

    def solve(self) -> SolveResult:
        """
        Run iterations until convergence or ``max_iter``.

        Returns
        -------
        SolveResult
            Final solver result and iteration history.
        """
        while not self.converged and not self.terminated and self.k < self.max_iter:
            self.step()

        return SolveResult(
            x=self.x.copy(),
            converged=self.converged,
            iters=self.k,
            func_evals=self.func_evals,
            history=self.history,
        )


def _print_history_vec(title: str, result: SolveResult) -> None:
    print(f"\n{title}")
    if result.x.size == 1:
        print("k        x                   g(x)-x      omega   beta")
        for k, xk, rk, _, omega, beta in result.history:
            print(f"{k:2d}  {float(xk[0]): .16f}  {float(rk[0]): .3e}  {omega: .3f}  {beta: .3f}")
    else:
        print("k        ||g(x)-x||_scaled    omega   beta")
        for k, _, __, rnorm, omega, beta in result.history:
            print(f"{k:2d}  {rnorm: .3e}  {omega: .3f}  {beta: .3f}")
    if result.x.size == 1:
        print(
            f"-> x = {float(result.x[0]):.16f}, converged={result.converged}, "
            f"iters={result.iters}, func_evals={result.func_evals}"
        )
    else:
        print(
            f"-> converged={result.converged}, iters={result.iters}, "
            f"func_evals={result.func_evals}, final ||g(x)-x||_scaled={result.history[-1][3]:.3e}"
        )


if __name__ == "__main__":
    g_vec = lambda x: np.cos(x)
    x0 = np.array([1.0], dtype=float)
    result = RobustFixedPointSolver(g_vec, x0, tol=1e-5, max_iter=50, verbose=False).solve()
    _print_history_vec("Robust Anderson-accelerated fixed point (class)", result)

    scipy_history: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
    scipy_func_evals = [0]
    scipy_jac_evals = [0]

    def F(x: np.ndarray) -> np.ndarray:
        fx = np.cos(x) - x
        rnorm = float(np.linalg.norm(fx) / (max(1, fx.size) ** 0.5))
        scipy_history.append((scipy_func_evals[0], x.copy(), fx.copy(), rnorm))
        scipy_func_evals[0] += 1
        return fx

    def J(x: np.ndarray) -> np.ndarray:
        scipy_jac_evals[0] += 1
        return np.diag(-np.sin(x) - 1.0)

    from scipy.optimize import root
    sol = root(F, x0.copy(), method="hybr", jac=J)

    print("\nSciPy root(method='hybr')")
    print("k        x                   g(x)-x")
    for k, xk, rk, _ in scipy_history:
        print(f"{k:2d}  {float(xk[0]): .16f}  {float(rk[0]): .3e}")
    print(
        f"-> x = {float(sol.x[0]):.16f}, converged={bool(sol.success)}, "
        f"iters={int(sol.nfev)}, func_evals={int(sol.nfev)}, jac_evals={scipy_jac_evals[0]}"
    )
