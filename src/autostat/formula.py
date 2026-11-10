"""The formula a structure search found, written as mathematics.

What this module does
    The structure search is a symbolic regression: it searches a grammar of
    expressions (sums, products and changes of regime over base kernels) for
    the one that explains the data best by BIC. This module writes the
    expression it kept as a formula, with every fitted parameter in place:

    * the model, ``y(t) = mu + f_1(t) + ... + f_n(t)``, one ``f_i`` per
      additive component, in the report's order;
    * each component's law, ``f_i ~ GP(0, k_i)``, with ``k_i(t, t')``
      written out in full from the fitted kernel;
    * for a component that is a polynomial (products of constants and linear
      kernels), the polynomial itself, read from its posterior mean, since
      a Gaussian process with such a covariance can only draw polynomials of
      that degree;
    * for a periodic component, its Fourier series, truncated where the
      remaining harmonics weigh less than :data:`FOURIER_TOLERANCE` of the
      component. The periodic kernel is exactly a sum of cosines,
      ``exp(a cos u) = I_0(a) + 2 sum_k I_k(a) cos(k u)``, so its posterior
      mean is a Fourier series in the period found, whose weights fall
      quickly when ``a`` is small;
    * either of the two switched on or off by one change of regime;
    * on request (``symbolic=True``, which needs PySR), for a smooth
      component, a formula found by symbolic regression on its posterior
      mean: a denoised one-dimensional curve. It is kept only when it
      reproduces the component to within :data:`SYMBOLIC_TOLERANCE`.

    Smooth components (squared exponential, rational quadratic,
    gamma-exponential) have no closed form beyond their covariance, nor do
    periodic ones modulated by them: their formula is the covariance, as in the
    original papers.

Why LaTeX
    The same strings go into the Markdown report (between ``$$``, which
    Pandoc and md2star typeset) and into the LaTeX report.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import kernels as ff
from .model import GPModel, posterior

#: Largest relative gap between a component's posterior mean and the
#: polynomial fitted to it for the polynomial to be called its formula.
POLYNOMIAL_TOLERANCE = 1e-6

#: Largest relative gap between a periodic component's posterior mean and
#: its truncated Fourier series for the series to be shown, with ``\approx``.
FOURIER_TOLERANCE = 1e-3

#: Most harmonics a Fourier closed form may use.
FOURIER_MAX_HARMONICS = 8

#: Largest relative gap between a smooth component's posterior mean and a
#: formula from symbolic regression for the formula to be shown.
SYMBOLIC_TOLERANCE = 0.01

#: How long one component's symbolic regression may run.
SYMBOLIC_TIMEOUT_SECONDS = 900


def number(value: float, digits: int = 4) -> str:
    """Write a number in LaTeX with a few significant digits.

    Parameters
    ----------
    value : float
        The number.
    digits : int, optional
        Significant digits.

    Returns
    -------
    str
        E.g. ``"0.8513"``, ``"1949"`` or ``"2.31 times 10^{-5}"`` in LaTeX.
    """
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    exponent = math.floor(math.log10(abs(value)))
    if -3 <= exponent < 6:
        text = f"{value:.{max(digits - 1 - exponent, 0)}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
    mantissa = value / 10**exponent
    text = f"{mantissa:.{digits - 1}f}".rstrip("0").rstrip(".")
    return rf"{text} \times 10^{{{exponent}}}"


#: Base kernels whose covariance carries an amplitude ``exp(2 sf)``.
_SCALED = (
    ff.NoiseKernel,
    ff.ConstKernel,
    ff.SqExpKernel,
    ff.RQKernel,
    ff.GammaExpKernel,
    ff.PeriodicKernel,
    ff.LinearKernel,
)


def _squared(text: str) -> str:
    """Square a number written in LaTeX, parenthesised when it has an exponent.

    A number written with a power of ten, followed by ``^2``, is a double
    superscript, which LaTeX refuses; in parentheses it is not.

    Parameters
    ----------
    text : str
        A number as :func:`number` writes it.

    Returns
    -------
    str
        The number squared, in LaTeX.
    """
    return rf"\left({text}\right)^2" if "^" in text else f"{text}^2"


def _shape_tex(kernel: ff.Kernel) -> str:
    """Write a base kernel's shape, its amplitude left out.

    Parameters
    ----------
    kernel : Kernel
        A fitted base kernel.

    Returns
    -------
    str
        LaTeX; ``""`` for a constant, whose shape is 1.

    Raises
    ------
    TypeError
        For a kernel class this module does not know.
    """
    if isinstance(kernel, ff.NoiseKernel):
        return r"\delta_{t t'}"
    if isinstance(kernel, ff.ConstKernel):
        return ""
    if isinstance(kernel, ff.SqExpKernel):
        ell = number(math.exp(kernel.lengthscale))
        return rf"\exp\!\left(-\frac{{(t - t')^2}}{{2 \cdot {_squared(ell)}}}\right)"
    if isinstance(kernel, ff.RQKernel):
        ell = number(math.exp(kernel.lengthscale))
        alpha = number(math.exp(kernel.alpha))
        return rf"\left(1 + \frac{{(t - t')^2}}{{2 \cdot {alpha} \cdot {_squared(ell)}}}\right)^{{-{alpha}}}"
    if isinstance(kernel, ff.GammaExpKernel):
        ell = number(math.exp(kernel.lengthscale))
        return rf"\exp\!\left(-\left(\frac{{|t - t'|}}{{{ell}}}\right)^{{{number(kernel.gamma)}}}\right)"
    if isinstance(kernel, ff.PeriodicKernel):
        a = number(1 / math.exp(2 * kernel.lengthscale))
        period = number(math.exp(kernel.period))
        return (
            rf"\frac{{\exp\!\left({a}\cos\frac{{2\pi (t - t')}}{{{period}}}\right) - I_0({a})}}"
            rf"{{e^{{{a}}} - I_0({a})}}"
        )
    if isinstance(kernel, ff.LinearKernel):
        c = number(kernel.location)
        return rf"(t - {c})(t' - {c})"
    raise TypeError(f"no formula for kernel {type(kernel).__name__}")


def kernel_tex(
    kernel: ff.Kernel, switches: dict[tuple[float, ...], str] | None = None
) -> str:
    """Write a fitted kernel as a covariance ``k(t, t')`` in LaTeX.

    A product's amplitudes are multiplied into one coefficient. A change of
    regime is written with a named weight, ``w_1(t)``, registered in
    *switches* so the caller can define it once.

    Parameters
    ----------
    kernel : Kernel
        A fitted kernel: every parameter set, logs as the package stores
        them.
    switches : dict, optional
        Weights already named, by their parameters; new ones are added.

    Returns
    -------
    str
        The covariance, numbers in place.

    Raises
    ------
    TypeError
        For a kernel class this module does not know.
    """
    switches = {} if switches is None else switches
    if isinstance(kernel, ff.ZeroKernel):
        return "0"
    if isinstance(kernel, _SCALED):
        shape = _shape_tex(kernel)
        scale = number(math.exp(2 * kernel.sf))
        return rf"{scale}\,{shape}" if shape else scale
    if isinstance(kernel, ff.SumKernel):
        return " + ".join(kernel_tex(k, switches) for k in kernel.operands)
    if isinstance(kernel, ff.ProductKernel):
        scale, factors = 1.0, []
        for k in kernel.operands:
            if isinstance(k, _SCALED):
                scale *= math.exp(2 * k.sf)
                shape = _shape_tex(k)
                if shape:
                    factors.append(shape)
            else:
                text = kernel_tex(k, switches)
                factors.append(rf"\left({text}\right)" if k.is_operator else text)
        body = r" \cdot ".join(factors)
        return rf"{number(scale)}\,{body}" if body else number(scale)
    if isinstance(kernel, ff.ChangePointKernel | ff.ChangeWindowKernel):
        key = (
            float(isinstance(kernel, ff.ChangeWindowKernel)),
            kernel.location,
            kernel.steepness,
            getattr(kernel, "width", 0.0) or 0.0,
        )
        name = switches.setdefault(key, f"w_{{{len(switches) + 1}}}")
        first, second = kernel.operands
        terms = []
        if not isinstance(first, ff.ZeroKernel):
            terms.append(
                rf"(1 - {name}(t))(1 - {name}(t'))\left[{kernel_tex(first, switches)}\right]"
            )
        if not isinstance(second, ff.ZeroKernel):
            terms.append(
                rf"{name}(t)\,{name}(t')\left[{kernel_tex(second, switches)}\right]"
            )
        return " + ".join(terms) or "0"
    raise TypeError(f"no formula for kernel {type(kernel).__name__}")


def switch_tex(key: tuple[float, ...], name: str) -> str:
    """Define a change-of-regime weight, as registered by :func:`kernel_tex`.

    Parameters
    ----------
    key : tuple of float
        ``(is_window, location, log steepness, log width)``.
    name : str
        Its symbol, e.g. ``w_{1}``.

    Returns
    -------
    str
        ``w_1(t) = ...``: the weight of the regime that starts at the change
        point, or that holds inside the window.
    """
    is_window, location, steepness, width = key
    steep = number(math.exp(steepness))
    if not is_window:
        return (
            rf"{name}(t) = \tfrac{{1}}{{2}}\left(1 + \tanh\left({steep}\,(t - "
            rf"{number(location)})\right)\right)"
        )
    half = math.exp(width) / 2
    start, end = number(location - half), number(location + half)
    return (
        rf"{name}(t) = \tfrac{{1}}{{4}}\left(1 + \tanh\left({steep}\,(t - {start})\right)\right)"
        rf"\left(1 - \tanh\left({steep}\,(t - {end})\right)\right)"
    )


def switch_weight(key: tuple[float, ...], t: np.ndarray) -> np.ndarray:
    """Evaluate a change-of-regime weight, as :func:`switch_tex` writes it.

    Parameters
    ----------
    key : tuple of float
        ``(is_window, location, log steepness, log width)``.
    t : numpy.ndarray
        Inputs.

    Returns
    -------
    numpy.ndarray
        Values in ``[0, 1]``.
    """
    is_window, location, steepness, width = key
    steep = math.exp(steepness)
    if not is_window:
        return 0.5 + 0.5 * np.tanh(steep * (t - location))
    half = math.exp(width) / 2
    rise = 0.5 + 0.5 * np.tanh(steep * (t - (location - half)))
    fall = 0.5 + 0.5 * np.tanh(-steep * (t - (location + half)))
    return rise * fall


def _shape(kernel: ff.Kernel) -> tuple[str, float] | None:
    """Say which closed form an unswitched component can have.

    Parameters
    ----------
    kernel : Kernel
        One additive component, or the live operand of a switch.

    Returns
    -------
    tuple or None
        ``("polynomial", degree)`` for products of constants and linear
        kernels; ``("fourier", period)`` for one periodic kernel, scaled by
        constants at most; None otherwise.
    """
    degree = polynomial_degree(kernel)
    if degree is not None:
        return "polynomial", float(degree)
    factors = kernel.operands if isinstance(kernel, ff.ProductKernel) else [kernel]
    periodic = [k for k in factors if isinstance(k, ff.PeriodicKernel)]
    if len(periodic) == 1 and all(
        isinstance(k, ff.ConstKernel | ff.PeriodicKernel) for k in factors
    ):
        return "fourier", math.exp(periodic[0].period)
    return None


def closed_form_shape(
    kernel: ff.Kernel, switches: dict[tuple[float, ...], str]
) -> tuple[str, float, str, tuple[float, ...] | None, bool] | None:
    """Recognise a polynomial or periodic component, switched or not.

    Parameters
    ----------
    kernel : Kernel
        One additive component.
    switches : dict
        The weights :func:`kernel_tex` named, by their parameters.

    Returns
    -------
    tuple or None
        ``(kind, value, prefix, key, inside)``: ``kind`` and ``value`` as
        :func:`_shape` gives them, the LaTeX factor in front (``""``,
        ``w_1(t)`` or ``(1 - w_1(t))``), the switch's key and whether the
        component holds where the weight is 1. None when the component has
        no closed form here.
    """
    shape = _shape(kernel)
    if shape is not None:
        return shape[0], shape[1], "", None, True
    if not isinstance(kernel, ff.ChangePointKernel | ff.ChangeWindowKernel):
        return None
    first, second = kernel.operands
    zero_first = isinstance(first, ff.ZeroKernel)
    zero_second = isinstance(second, ff.ZeroKernel)
    if zero_first == zero_second:
        return None
    shape = _shape(second if zero_first else first)
    if shape is None:
        return None
    key = (
        float(isinstance(kernel, ff.ChangeWindowKernel)),
        kernel.location,
        kernel.steepness,
        getattr(kernel, "width", 0.0) or 0.0,
    )
    name = switches.get(key, "w")
    prefix = rf"{name}(t)\," if zero_first else rf"(1 - {name}(t))\,"
    return shape[0], shape[1], prefix, key, zero_first


def fourier_tex(coefficients: np.ndarray, period: float, origin: float) -> str:
    """Write a Fourier series in amplitude-phase form.

    Parameters
    ----------
    coefficients : numpy.ndarray
        ``[a_1, b_1, a_2, b_2, ...]``: cosine and sine weights per harmonic.
    period : float
        The period, in the input's unit.
    origin : float
        Where the phase is measured from.

    Returns
    -------
    str
        ``A_1 cos(2 pi (t - o) / p - phi_1) + ...`` in LaTeX.
    """
    variable = rf"(t - {number(origin)})" if origin else "t"
    terms = []
    for k in range(len(coefficients) // 2):
        a, b = coefficients[2 * k], coefficients[2 * k + 1]
        amplitude = math.hypot(a, b)
        phase = math.atan2(b, a)
        harmonic = "" if k == 0 else f"{k + 1} \\cdot "
        angle = rf"\frac{{2\pi \cdot {harmonic}{variable}}}{{{number(period)}}}"
        shift = (
            f" - {number(phase)}"
            if phase > 0
            else f" + {number(-phase)}"
            if phase < 0
            else ""
        )
        terms.append(rf"{number(amplitude)}\cos\!\left({angle}{shift}\right)")
    return " + ".join(terms)


def is_smooth(kernel: ff.Kernel) -> bool:
    """Say whether a component is a smooth curve symbolic regression can read.

    Parameters
    ----------
    kernel : Kernel
        One additive component.

    Returns
    -------
    bool
        True for a product of smooth, linear and constant kernels with at
        least one smooth factor. Noise has no curve to read; a periodic
        factor modulated by a smooth one has too many cycles for a short
        formula.
    """
    factors = kernel.operands if isinstance(kernel, ff.ProductKernel) else [kernel]
    smooth = (ff.SqExpKernel, ff.RQKernel, ff.GammaExpKernel)
    allowed = (*smooth, ff.LinearKernel, ff.ConstKernel)
    return all(isinstance(k, allowed) for k in factors) and any(
        isinstance(k, smooth) for k in factors
    )


def symbolic_form(u: np.ndarray, curve: np.ndarray) -> tuple[str, float] | None:
    """Find a short formula for a curve with PySR, if one is close enough.

    PySR runs in a process of its own (``_pysr_worker.py``): Julia and
    ``torch`` do not share one safely.

    Parameters
    ----------
    u : numpy.ndarray
        Inputs, shape ``(m,)``.
    curve : numpy.ndarray
        The curve to read, shape ``(m,)``.

    Returns
    -------
    tuple or None
        The formula in LaTeX (variable ``u``) with its largest gap as
        a fraction of the curve's largest value; None when no formula is
        within :data:`SYMBOLIC_TOLERANCE`. A warning says so when PySR is
        missing or fails.
    """
    request = {
        "u": u.tolist(),
        "curve": curve.tolist(),
        "tolerance": SYMBOLIC_TOLERANCE,
    }
    worker = Path(__file__).with_name("_pysr_worker.py")
    try:
        done = subprocess.run(
            [sys.executable, str(worker)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=SYMBOLIC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        warnings.warn(
            "PySR timed out: the component keeps its covariance", stacklevel=2
        )
        return None
    if done.returncode == 3:
        warnings.warn(
            "PySR not found (pip install 'automatic-statistician-revisited[symbolic]'): "
            "smooth components keep their covariance as their formula",
            stacklevel=2,
        )
        return None
    if done.returncode != 0:
        warnings.warn(
            f"PySR failed (status {done.returncode}): the component keeps its covariance",
            stacklevel=2,
        )
        return None
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    return None if answer["latex"] is None else (answer["latex"], float(answer["gap"]))


def polynomial_degree(kernel: ff.Kernel) -> int | None:
    """Give the degree of the polynomials a kernel draws, if it draws only those.

    Parameters
    ----------
    kernel : Kernel
        One additive component.

    Returns
    -------
    int or None
        The number of linear factors for a product of constants and linear
        kernels (a lone constant is degree 0); None for anything else.
    """
    factors = kernel.operands if isinstance(kernel, ff.ProductKernel) else [kernel]
    if not all(isinstance(k, ff.ConstKernel | ff.LinearKernel) for k in factors):
        return None
    return sum(isinstance(k, ff.LinearKernel) for k in factors)


def polynomial_tex(coefficients: np.ndarray, origin: float) -> str:
    """Write a polynomial in powers of ``t - origin``.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Constant term first.
    origin : float
        Where the powers are taken from.

    Returns
    -------
    str
        LaTeX.
    """
    variable = rf"(t - {number(origin)})" if origin else "t"
    terms: list[str] = []
    for power, value in enumerate(coefficients):
        if value == 0:
            continue
        magnitude = number(abs(value))
        if power == 0:
            body = magnitude
        elif power == 1:
            body = rf"{magnitude}\,{variable}"
        else:
            body = rf"{magnitude}\,{variable}^{{{power}}}"
        if not terms:
            terms.append(f"-{body}" if value < 0 else body)
        else:
            terms.append(f"- {body}" if value < 0 else f"+ {body}")
    return " ".join(terms) or "0"


@dataclass
class Formula:
    """The model as mathematics, ready for a report.

    Parameters
    ----------
    model : str
        ``y(t) = ...`` with one ``f_i(t)`` per component.
    laws : list of str
        ``f_i(t) ~ GP(0, k_i)`` with ``k_i`` written out.
    explicit : list of str or None
        Per component, its polynomial when it is one, else None.
    noise : str or None
        The likelihood's noise, when the model has one outside its kernel.
    switches : list of str
        Definitions of the change-of-regime weights the laws use.
    unit : str
        What ``t`` is measured in.
    notes : list of str or None
        Per component, how its closed form was obtained and how close it is
        (``"exact polynomial of degree 1"``, ``"Fourier series, 2 harmonics,
        largest gap 0.03 % of the component"``); None without one.
    """

    model: str
    laws: list[str]
    explicit: list[str | None]
    noise: str | None
    switches: list[str]
    unit: str
    notes: list[str | None] = field(default_factory=list)


def formula(
    model: GPModel,
    parts: list[ff.Kernel],
    order: list[int],
    x: np.ndarray,
    y: np.ndarray,
    unit: str = "year",
    symbolic: bool = False,
) -> Formula:
    """Write a fitted model, component by component, as mathematics.

    Parameters
    ----------
    model : GPModel
        The fitted (simplified) model.
    parts : list of Kernel
        Its additive components.
    order : list of int
        Report order of ``parts``.
    x, y : numpy.ndarray
        The data, for the polynomial components' posterior means.
    unit : str, optional
        ``"year"`` or ``""``.
    symbolic : bool, optional
        Look for a formula for smooth components with PySR.

    Returns
    -------
    Formula
        The model, the laws and the explicit polynomials.
    """
    names = [f"f_{{{i + 1}}}(t)" for i in range(len(order))]
    mean = f"{number(model.mean_const)} + " if model.mean_const else ""
    noise = None
    if math.isfinite(model.lik_sf):
        noise = rf"\varepsilon(t) \sim \mathrm{{N}}\!\left(0,\ {number(math.exp(2 * model.lik_sf))}\right)"
    model_tex = (
        "y(t) = " + mean + " + ".join(names) + (r" + \varepsilon(t)" if noise else "")
    )
    lo, hi = float(x[:, 0].min()), float(x[:, 0].max())
    origin = float(round(lo)) if unit == "year" else 0.0
    grid = np.linspace(lo, hi, 200)[:, None]
    laws, explicit, notes = [], [], []
    switches: dict[tuple[float, ...], str] = {}
    for name, j in zip(names, order, strict=True):
        part = parts[j]
        covariance = kernel_tex(part, switches)
        laws.append(rf"{name} \sim \mathrm{{GP}}\!\left(0,\ {covariance}\right)")
        shape = closed_form_shape(part, switches)
        if shape is None:
            found = None
            if symbolic and is_smooth(part):
                t = grid[:, 0]
                curve, _ = posterior(model, x, y, grid, component=part)
                found = symbolic_form(t - origin, curve)
            if found is None:
                explicit.append(None)
                notes.append(None)
                continue
            where = rf"\quad u = t - {number(origin)}" if origin else r"\quad u = t"
            explicit.append(rf"{name} \approx {found[0]},{where}")
            notes.append(
                f"symbolic regression (PySR) on the posterior mean, largest gap "
                f"{100 * found[1]:.2g} % of the component"
            )
            continue
        kind, value, prefix, key, inside = shape
        t = grid[:, 0]
        if kind == "fourier":
            per_period = 40 * (hi - lo) / value
            t = np.linspace(lo, hi, int(min(max(200, per_period), 4000)))
        mean_j, _ = posterior(model, x, y, t[:, None], component=part)
        weight = np.ones_like(t)
        if key is not None:
            weight = switch_weight(key, t) if inside else 1 - switch_weight(key, t)
        scale = max(float(np.max(np.abs(mean_j))), 1e-12)
        if kind == "polynomial":
            design = weight[:, None] * (t - origin)[:, None] ** np.arange(
                int(value) + 1
            )
            coefficients, *_ = np.linalg.lstsq(design, mean_j, rcond=None)
            gap = float(np.max(np.abs(design @ coefficients - mean_j)))
            if gap > POLYNOMIAL_TOLERANCE * scale:
                explicit.append(None)
                notes.append(None)
                continue
            body, sign, degree = polynomial_tex(coefficients, origin), "=", int(value)
            notes.append(f"exact polynomial of degree {degree}")
        else:
            body = None
            for harmonics in range(1, FOURIER_MAX_HARMONICS + 1):
                k = np.arange(1, harmonics + 1)
                angle = 2 * np.pi * np.outer(t - origin, k) / value
                design = np.stack([np.cos(angle), np.sin(angle)], axis=2)
                design = weight[:, None] * design.reshape(len(t), -1)
                coefficients, *_ = np.linalg.lstsq(design, mean_j, rcond=None)
                gap = float(np.max(np.abs(design @ coefficients - mean_j)))
                if gap <= FOURIER_TOLERANCE * scale:
                    body = fourier_tex(coefficients, value, origin)
                    plural = "s" if harmonics > 1 else ""
                    notes.append(
                        f"Fourier series, {harmonics} harmonic{plural}, largest gap "
                        f"{100 * gap / scale:.2g} % of the component"
                    )
                    break
            if body is None:
                explicit.append(None)
                notes.append(None)
                continue
            sign, degree = r"\approx", 1
        if prefix and kind == "polynomial" and degree == 0:
            explicit.append(rf"{name} {sign} {body}\,{prefix.rstrip(chr(92) + ',')}")
        elif prefix:
            explicit.append(rf"{name} {sign} {prefix}\left({body}\right)")
        else:
            explicit.append(rf"{name} {sign} {body}")
    definitions = [switch_tex(key, name) for key, name in switches.items()]
    return Formula(model_tex, laws, explicit, noise, definitions, unit, notes)
