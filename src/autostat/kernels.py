"""Compositional covariance functions: the language the structure search speaks.

What this module does
    It defines every covariance function (kernel) the search combines: the
    base kernels (smooth, periodic, linear, constant, white noise, rational
    quadratic) and the operators that combine them (sum, product, changepoint,
    change window). Each kernel object holds its structure as Python objects
    and evaluates its covariance matrix as a ``torch`` expression.

Why it exists as a separate module
    The search, the optimiser and the report all manipulate kernel
    *expressions*: they expand them, simplify them, compare them and split
    them into additive parts. That algebra is independent of how a model is
    fitted, so it lives on its own.

What it consumes
    Inputs as ``torch`` tensors of shape ``(n, D)`` and parameter vectors in
    the order the original GPML toolbox used.

What it produces
    Covariance matrices, the structural operations (canonical form,
    simplification, additive form, summands) and a JSON round trip.

Provenance
    A port of the kernel half of ``flexible_function.py`` from
    ``github.com/jamesrobertlloyd/gpss-research`` (Lloyd, Duvenaud, Grosse,
    Tenenbaum and Ghahramani, AAAI 2014) together with the MATLAB covariance
    functions of the GPML (Gaussian Processes for Machine Learning) toolbox
    that the original compiled them to.

    The parameterisation is GPML's, number for number: log lengthscales, log
    scale factors, log periods and raw locations, in GPML order. A kernel
    printed by the 2014 code therefore means the same thing here, which is
    what lets the published results serve as a regression test. Gradients
    come from autograd instead of GPML's hand-written derivatives.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>. The ported design is (c) 2013-2015
James Robert Lloyd, David Duvenaud and Roger Grosse, MIT licence, reproduced
in ``LICENSE-ORIGINAL``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import torch

Tensor = torch.Tensor

#: Above this lengthscale the periodic kernel is evaluated at its limit,
#: ``cos(2 pi r / p)``, as GPML's ``covPeriodicNoDC`` did. The closed form
#: cancels catastrophically there.
PERIODIC_ELL_LIMIT = 1e4


def _fmt(value: float | None) -> str:
    """Format one parameter as the original pretty-printer did.

    Parameters
    ----------
    value : float or None
        A parameter, unset parameters being ``None``.

    Returns
    -------
    str
        One decimal, or ``"None"``.
    """
    return "None" if value is None else f"{value:1.1f}"


def _column(x: Tensor, dimension: int | None) -> Tensor:
    """Select the input dimension a kernel acts on.

    Parameters
    ----------
    x : torch.Tensor
        Inputs, shape ``(n, D)``.
    dimension : int or None
        Column index; ``None`` means the first column.

    Returns
    -------
    torch.Tensor
        Shape ``(n,)``.
    """
    return x[:, dimension if dimension is not None else 0]


def _difference(x: Tensor, z: Tensor, dimension: int | None) -> Tensor:
    """Compute pairwise signed differences along one input dimension.

    Parameters
    ----------
    x, z : torch.Tensor
        Inputs, shapes ``(n, D)`` and ``(m, D)``.
    dimension : int or None
        Column index.

    Returns
    -------
    torch.Tensor
        Shape ``(n, m)``, entry ``(i, j)`` being ``x_i - z_j``.
    """
    return _column(x, dimension)[:, None] - _column(z, dimension)[None, :]


def as_input(x: Any) -> Tensor:
    """Convert array-like inputs to a float64 tensor of shape ``(n, D)``.

    Parameters
    ----------
    x : array_like
        One-dimensional inputs become a single column.

    Returns
    -------
    torch.Tensor
        Shape ``(n, D)``, dtype float64.
    """
    t = torch.as_tensor(np.asarray(x, float), dtype=torch.float64)
    return t[:, None] if t.ndim == 1 else t


class Kernel:
    """Base class for every covariance function, base or operator.

    Parameters live on the instance as plain floats, ``None`` until
    initialised, under the names listed in :attr:`param_names`. A covariance
    is evaluated by :meth:`cov` from a flat tensor slice, so the optimiser
    owns the numbers while the structure stays a Python object.
    """

    #: Parameter names, in GPML order.
    param_names: ClassVar[tuple[str, ...]] = ()
    #: Identifier used in kernel lists such as ``"SE,Per,Lin"``.
    id: ClassVar[str] = "Kernel"
    #: A thunk takes no input dimension (noise, constant, zero).
    is_thunk: ClassVar[bool] = False
    is_operator: ClassVar[bool] = False
    is_abelian: ClassVar[bool | None] = None
    arity: ClassVar[int | str] = 0

    dimension: int | None = None
    operands: list[Kernel]

    # Parameters

    @property
    def param_vector(self) -> np.ndarray:
        """All parameters in GPML order.

        Returns
        -------
        numpy.ndarray
            Float vector of length :attr:`num_params`.
        """
        return np.array([getattr(self, name) for name in self.param_names], float)

    @property
    def num_params(self) -> int:
        """Length of :attr:`param_vector`.

        Returns
        -------
        int
            Parameter count, operands included.
        """
        return len(self.param_names)

    def load_param_vector(self, params: Any) -> None:
        """Set every parameter from a vector in GPML order.

        Parameters
        ----------
        params : array_like
            Exactly :attr:`num_params` values.

        Raises
        ------
        ValueError
            If the length is wrong.
        """
        values = list(np.asarray(params, float).ravel())
        for name, value in zip(self.param_names, values, strict=True):
            setattr(self, name, float(value))

    @property
    def effective_params(self) -> int:
        """Parameters counted by the information criteria.

        Returns
        -------
        int
            For a base kernel, its parameter count.
        """
        return self.num_params

    @property
    def depth(self) -> int:
        """Nesting depth of the expression.

        Returns
        -------
        int
            Zero for a base kernel.
        """
        return 0

    @property
    def is_stationary(self) -> bool:
        """Whether the covariance depends on inputs only through their difference.

        Returns
        -------
        bool
            True unless overridden.
        """
        return True

    @property
    def sf(self) -> float:
        """Log scale factor, the log of the marginal standard deviation.

        Returns
        -------
        float
            The scale factor.

        Raises
        ------
        RuntimeError
            For kernels without one.
        """
        raise RuntimeError(f"{self.id} has no scale factor")

    def multiply_by_const(self, sf: float) -> None:
        """Absorb a multiplicative constant ``exp(2 sf)`` into the scale.

        Parameters
        ----------
        sf : float
            Log standard deviation of the constant.

        Raises
        ------
        RuntimeError
            For kernels without a scale factor.
        """
        raise RuntimeError(f"{self.id} has no scale factor to absorb a constant")

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw every unset parameter from its data-dependent heuristic.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread of the random restarts, on the log scale.
        shape : dict
            Data summary built by :func:`autostat.search.data_shape`.
        """

    def out_of_bounds(self, shape: dict[str, Any]) -> bool:
        """Whether optimisation drove the kernel outside its sensible range.

        Parameters
        ----------
        shape : dict
            Data summary built by :func:`autostat.search.data_shape`.

        Returns
        -------
        bool
            False unless a subclass sets bounds.
        """
        return False

    # Evaluation

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Evaluate the covariance matrix between two sets of inputs.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs, shapes ``(n, D)`` and ``(m, D)``.
        theta : torch.Tensor
            This kernel's parameters, in :attr:`param_vector` order.
        same : bool
            Whether ``x`` and ``z`` are the same points. Only white noise
            reads it: the noise is the identity on a set with itself and zero
            between distinct sets.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.

        Raises
        ------
        NotImplementedError
            Always: every concrete kernel overrides it.
        """
        raise NotImplementedError

    def __call__(self, x: Any, z: Any = None) -> np.ndarray:
        """Evaluate at the stored parameters, for inspection and plotting.

        Parameters
        ----------
        x : array_like
            Inputs.
        z : array_like, optional
            Second set of inputs; defaults to ``x`` itself.

        Returns
        -------
        numpy.ndarray
            The covariance matrix.
        """
        xt = as_input(x)
        same = z is None
        zt = xt if same else as_input(z)
        theta = torch.as_tensor(self.param_vector, dtype=torch.float64)
        with torch.no_grad():
            return self.cov(xt, zt, theta, same).numpy()

    # Identity and order

    def copy(self) -> Kernel:
        """Return a deep copy.

        Returns
        -------
        Kernel
            Shares no mutable state with ``self``.

        Raises
        ------
        NotImplementedError
            Always: every concrete kernel overrides it.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Render the constructor call that rebuilds this kernel.

        Returns
        -------
        str
            The original's format, e.g. ``SqExpKernel(dimension=0, ...)``.
        """
        parts = [] if self.is_thunk else [f"dimension={self.dimension}"]
        parts += [f"{name}={getattr(self, name)!r}" for name in self.param_names]
        return f"{type(self).__name__}({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        """Compare structure and parameters through the representation.

        Parameters
        ----------
        other : object
            Anything.

        Returns
        -------
        bool
            True when both representations agree.
        """
        return isinstance(other, Kernel) and repr(self) == repr(other)

    def __lt__(self, other: Kernel) -> bool:
        """Order kernels by class name, then representation.

        The original relied on Python 2's ``__cmp__``. This total order sorts
        the operands of sums and products into a canonical form.

        Parameters
        ----------
        other : Kernel
            Kernel to compare with.

        Returns
        -------
        bool
            True when ``self`` sorts first.
        """
        mine = (type(self).__name__, repr(self))
        theirs = (type(other).__name__, repr(other))
        return mine < theirs

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`.

        Returns
        -------
        int
            Hash of the representation.
        """
        return hash(repr(self))

    def pretty_print(self) -> str:
        """Render a short form with parameters to one decimal.

        Returns
        -------
        str
            E.g. ``SE(dim=0, lengthscale=1.2, sf=0.3)``.
        """
        parts = [] if self.is_thunk else [f"dim={self.dimension}"]
        parts += [f"{name}={_fmt(getattr(self, name))}" for name in self.param_names]
        return f"{self.id}({', '.join(parts)})"

    def structure(self) -> str:
        """Render the expression without parameters.

        Returns
        -------
        str
            E.g. ``WN + SE + Per x SE``.
        """
        return {"Noise": "WN", "Const": "C"}.get(self.id, self.id)

    def to_dict(self) -> dict[str, Any]:
        """Describe the kernel as JSON-serialisable data.

        Returns
        -------
        dict
            Inverted by :func:`from_dict`.
        """
        out: dict[str, Any] = {"type": type(self).__name__}
        if not self.is_thunk:
            out["dimension"] = self.dimension
        out.update({name: getattr(self, name) for name in self.param_names})
        return out

    # Arithmetic

    def __add__(self, other: Kernel) -> Kernel:
        """Build the sum of two kernels.

        Parameters
        ----------
        other : Kernel
            Second summand.

        Returns
        -------
        Kernel
            A flat :class:`SumKernel` of copies.
        """
        if not isinstance(other, Kernel):
            return NotImplemented
        left = self.operands if isinstance(self, SumKernel) else [self]
        return SumKernel([*(o.copy() for o in left), other.copy()])

    def __mul__(self, other: Kernel) -> Kernel:
        """Build the product of two kernels.

        Parameters
        ----------
        other : Kernel
            Second factor.

        Returns
        -------
        Kernel
            A flat :class:`ProductKernel` of copies.
        """
        if not isinstance(other, Kernel):
            return NotImplemented
        left = self.operands if isinstance(self, ProductKernel) else [self]
        return ProductKernel([*(o.copy() for o in left), other.copy()])

    # Structural algebra

    def canonical(self) -> Kernel:
        """Flatten nested sums and products and sort commutative operands.

        Returns
        -------
        Kernel
            Equal covariance, canonical expression. Deleted sub-expressions
            (:class:`NoneKernel`) disappear.
        """
        if not self.is_operator:
            return self.copy()
        new_ops: list[Kernel] = []
        for op in self.operands:
            op_canon = op.canonical()
            if isinstance(op_canon, type(self)) and self.arity == "n":
                new_ops += op_canon.operands
            elif not isinstance(op_canon, NoneKernel):
                new_ops.append(op_canon)
        if not new_ops:
            return NoneKernel()
        if len(new_ops) == 1:
            return new_ops[0]
        canon = self.copy()
        canon.operands = sorted(new_ops) if self.is_abelian else new_ops
        return canon

    def simplified(self) -> Kernel:
        """Apply the idempotency, identity and zero rules until nothing changes.

        Returns
        -------
        Kernel
            Equal covariance, fewer redundant parameters.
        """
        k = self.copy()
        previous = None
        while previous != k:
            previous = k.copy()
            k = k.collapse_additive_idempotency()
            k = k.collapse_multiplicative_idempotency()
            k = k.collapse_multiplicative_identity()
            k = k.collapse_multiplicative_zero()
            k = k.canonical()
        return k

    def additive_form(self) -> Kernel:
        """Rewrite as a sum of products with changepoints moved to the top.

        Returns
        -------
        Kernel
            Canonical, equal covariance.
        """
        k = self.canonical()
        if not isinstance(k, ProductKernel):
            if k.is_operator:
                rebuilt = k.copy()
                rebuilt.operands = [op.additive_form() for op in k.operands]
                return rebuilt.canonical()
            return k.canonical()

        switching = (ChangePointKernel, ChangeWindowKernel)
        factors = sorted(op.additive_form() for op in k.operands)
        result = factors[0]
        for factor in factors[1:]:
            # Changepoint operators take priority over sums: the product is
            # pushed inside them so that every summand of the final form says
            # where it acts. Sums come next, then plain multiplication.
            if isinstance(result, switching):
                result.operands = [
                    (op * factor.copy()).canonical().additive_form()
                    for op in result.operands
                ]
            elif isinstance(factor, switching):
                inner = result.copy()
                result = factor
                result.operands = [
                    (op * inner.copy()).canonical().additive_form()
                    for op in result.operands
                ]
            elif isinstance(result, SumKernel):
                result.operands = [
                    (op * factor.copy()).canonical().additive_form()
                    for op in result.operands
                ]
            elif isinstance(factor, SumKernel):
                inner = result.copy()
                result = factor
                result.operands = [
                    (op * inner.copy()).canonical().additive_form()
                    for op in result.operands
                ]
            else:
                result = result * factor
            result = result.canonical()
        return result

    def break_into_summands(self) -> list[Kernel]:
        """Distribute every product and list the resulting terms.

        Returns
        -------
        list of Kernel
            Terms whose sum is this kernel.
        """
        distributed = self.copy().distribute_products()
        if isinstance(distributed, SumKernel):
            return list(distributed.operands)
        return [distributed]

    def distribute_products(self) -> Kernel:
        """Expand products over sums, recursively, into a polynomial.

        Returns
        -------
        Kernel
            A :class:`SumKernel` of products, or a base kernel.

        Raises
        ------
        RuntimeError
            For an operator whose arity is neither two nor ``"n"``.
        """
        k = self.copy()
        if isinstance(k, ProductKernel):
            distributed = [op.break_into_summands() for op in k.operands]
            return SumKernel(
                [ProductKernel(list(term)) for term in itertools.product(*distributed)]
            )
        if isinstance(k, SumKernel):
            return SumKernel(
                [sub for op in k.operands for sub in op.break_into_summands()]
            )
        if not k.is_operator:
            return k
        if k.arity != 2:
            raise RuntimeError("cannot distribute products over this operator")
        # A changepoint of two sums is the sum of changepoints that each keep
        # one side and zero the other, so every term acts on one region.
        summands: list[Kernel] = []
        for op in k.operands[0].break_into_summands():
            term = k.copy()
            term.operands = [op, ZeroKernel()]
            summands.append(term)
        for op in k.operands[1].break_into_summands():
            term = k.copy()
            term.operands = [ZeroKernel(), op]
            summands.append(term)
        return SumKernel(summands)

    def collapse_additive_idempotency(self) -> Kernel:
        """Merge ``WN + WN`` into one noise and ``C + C`` into one constant.

        Returns
        -------
        Kernel
            Variances summed.
        """
        k = self.copy().canonical()
        if not k.is_operator:
            return k
        k.operands = [o.collapse_additive_idempotency() for o in k.operands]
        if isinstance(k, SumKernel):
            for cls in (NoiseKernel, ConstKernel):
                matches = [op for op in k.operands if isinstance(op, cls)]
                if matches:
                    variance = sum(math.exp(2 * op.sf) for op in matches)
                    others = [op for op in k.operands if not isinstance(op, cls)]
                    k.operands = [*others, cls(sf=0.5 * math.log(variance))]
            return k.canonical()
        return k

    def collapse_multiplicative_idempotency(self) -> Kernel:
        """Merge ``SE x SE``, ``WN x WN`` and ``C x C`` into one factor each.

        Returns
        -------
        Kernel
            Lengthscales combined in quadrature, log scales added.
        """
        k = self.copy().canonical()
        if not k.is_operator:
            return k
        ops = [o.collapse_multiplicative_idempotency() for o in k.operands]
        if not isinstance(k, ProductKernel):
            k.operands = ops
            return k
        lengthscales: dict[int | None, float] = {}
        scales: dict[int | None, float] = {}
        others: list[Kernel] = []
        for op in ops:
            if isinstance(op, SqExpKernel):
                previous = lengthscales.get(op.dimension, math.inf)
                # exp(-r^2/2a^2) exp(-r^2/2b^2) is an SE with 1/l^2 = 1/a^2 + 1/b^2.
                lengthscales[op.dimension] = -0.5 * math.log(
                    math.exp(-2 * previous) + math.exp(-2 * op.lengthscale)
                )
                scales[op.dimension] = scales.get(op.dimension, 0.0) + op.sf
            else:
                others.append(op)
        ops = others + [
            SqExpKernel(dimension=d, lengthscale=lengthscales[d], sf=scales[d])
            for d in lengthscales
        ]
        for cls in (NoiseKernel, ConstKernel):
            matches = [op for op in ops if isinstance(op, cls)]
            if matches:
                rest = [op for op in ops if not isinstance(op, cls)]
                ops = [*rest, cls(sf=sum(op.sf for op in matches))]
        k.operands = ops
        return k.canonical()

    def collapse_multiplicative_zero(self) -> Kernel:
        """Reduce white noise times anything stationary to white noise.

        Returns
        -------
        Kernel
            Non-stationary factors (such as ``Lin``) survive the product.
        """
        k = self.copy().canonical()
        if not k.is_operator:
            return k
        ops = [o.collapse_multiplicative_zero() for o in k.operands]
        if not isinstance(k, ProductKernel):
            k.operands = ops
            return k
        if any(isinstance(op, NoiseKernel) for op in ops):
            scale = sum(op.sf for op in ops if op.is_stationary)
            kept = [op for op in ops if not op.is_stationary]
            ops = [*kept, NoiseKernel(sf=scale)]
        k.operands = ops
        return k.canonical()

    def collapse_multiplicative_identity(self) -> Kernel:
        """Turn a constant factor into a scale on another factor.

        Returns
        -------
        Kernel
            One parameter fewer per absorbed constant.
        """
        k = self.copy().canonical()
        if not k.is_operator:
            return k
        ops = [o.collapse_multiplicative_identity() for o in k.operands]
        if not isinstance(k, ProductKernel):
            k.operands = ops
            return k
        consts = [op for op in ops if isinstance(op, ConstKernel)]
        if consts:
            ops = [op for op in ops if not isinstance(op, ConstKernel)]
            ops[0].multiply_by_const(sf=sum(op.sf for op in consts))
        k.operands = ops
        return k.canonical()

    def cp_structure(self) -> Kernel:
        """Replace base kernels by unit constants, keeping changepoint masks.

        Returns
        -------
        Kernel
            Its diagonal is the *envelope*: where the component is switched on.
        """
        k = self.copy()
        if isinstance(k, (ZeroKernel, NoneKernel)):
            return k
        if not k.is_operator:
            return ConstKernel(sf=0.0)
        k.operands = [op.cp_structure() for op in k.operands]
        return k


# Base kernels


class _BaseKernel(Kernel):
    """Shared constructor, copy and scale handling for base kernels."""

    def __init__(self, dimension: int | None = None, **params: float | None) -> None:
        """Create a base kernel with some or all parameters set.

        Parameters
        ----------
        dimension : int, optional
            Input dimension; ignored by thunks.
        **params : float or None
            Values for names in :attr:`param_names`; missing ones stay unset.

        Raises
        ------
        TypeError
            On an unknown parameter name.
        """
        self.dimension = None if self.is_thunk else dimension
        for name in self.param_names:
            setattr(self, name, params.pop(name, None))
        if params:
            raise TypeError(f"unknown parameters for {self.id}: {sorted(params)}")

    def copy(self) -> Kernel:
        """Return an independent copy with the same parameters.

        Returns
        -------
        Kernel
            Same class, same values.
        """
        params = {name: getattr(self, name) for name in self.param_names}
        return type(self)(dimension=self.dimension, **params)

    @property
    def sf(self) -> float:
        """Stored log scale factor.

        Returns
        -------
        float
            The ``sf`` parameter.
        """
        return self.__dict__["sf"]

    @sf.setter
    def sf(self, value: float) -> None:
        """Store the log scale factor.

        Parameters
        ----------
        value : float
            New log scale factor.
        """
        self.__dict__["sf"] = value

    def multiply_by_const(self, sf: float) -> None:
        """Add ``sf`` to the log scale factor.

        Parameters
        ----------
        sf : float
            Log standard deviation of the absorbed constant.

        Raises
        ------
        RuntimeError
            For a kernel without a scale factor.
        """
        if "sf" not in self.param_names:
            raise RuntimeError(f"{self.id} has no scale factor to absorb a constant")
        self.sf = self.sf + sf


def _output_scale(rng: np.random.Generator, sd: float, shape: dict[str, Any]) -> float:
    """Draw a log scale factor near the data's scale or near one.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    sd : float
        Spread on the log scale.
    shape : dict
        Data summary.

    Returns
    -------
    float
        A log scale factor.
    """
    return rng.normal(shape["y_sd"] if rng.random() < 0.5 else 0.0, sd)


def _input_lengthscale(
    rng: np.random.Generator, sd: float, shape: dict[str, Any], d: int | None
) -> float:
    """Draw a log lengthscale near the input spread or near twice its range.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    sd : float
        Spread on the log scale.
    shape : dict
        Data summary.
    d : int or None
        Input dimension.

    Returns
    -------
    float
        A log lengthscale; the long one is close to "no variation".
    """
    d = d or 0
    if rng.random() < 0.5:
        return rng.normal(shape["x_sd"][d], sd)
    return rng.normal(math.log(2 * (shape["x_max"][d] - shape["x_min"][d])), sd)


class ZeroKernel(_BaseKernel):
    """The zero covariance. It appears inside decomposed changepoints."""

    id = "Zero"
    is_thunk = True

    @property
    def sf(self) -> float:
        """Log scale of zero.

        Returns
        -------
        float
            Minus infinity.
        """
        return -math.inf

    def multiply_by_const(self, sf: float) -> None:
        """Leave zero unchanged: a scaled zero is zero.

        Parameters
        ----------
        sf : float
            Ignored.
        """

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Return zeros.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            Empty.
        same : bool
            Ignored.

        Returns
        -------
        torch.Tensor
            Zeros of shape ``(n, m)``.
        """
        return torch.zeros(x.shape[0], z.shape[0], dtype=torch.float64)

    def __repr__(self) -> str:
        """Render ``ZeroKernel()``.

        Returns
        -------
        str
            The constructor call.
        """
        return "ZeroKernel()"


class NoneKernel(_BaseKernel):
    """Placeholder the grammar uses to delete a sub-expression."""

    id = "None"
    is_thunk = True

    def multiply_by_const(self, sf: float) -> None:
        """Leave the placeholder unchanged.

        Parameters
        ----------
        sf : float
            Ignored.
        """

    def __repr__(self) -> str:
        """Render ``NoneKernel()``.

        Returns
        -------
        str
            The constructor call.
        """
        return "NoneKernel()"


class NoiseKernel(_BaseKernel):
    """White noise (``WN``): ``sf^2`` on the diagonal of a set with itself."""

    id = "Noise"
    is_thunk = True
    param_names = ("sf",)

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw the noise scale near a tenth of the data's or near one.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        if self.sf is None:
            loc = shape["y_sd"] - math.log(10) if rng.random() < 0.5 else 0.0
            self.sf = rng.normal(loc, sd)

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Return ``sf^2 I`` on a set with itself, zeros otherwise.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            ``[sf]``.
        same : bool
            Whether ``x`` and ``z`` are the same points.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        if not same:
            return torch.zeros(x.shape[0], z.shape[0], dtype=torch.float64)
        return torch.exp(2 * theta[0]) * torch.eye(x.shape[0], dtype=torch.float64)


class ConstKernel(_BaseKernel):
    """Constant covariance ``sf^2`` (``C``): a random offset."""

    id = "Const"
    is_thunk = True
    param_names = ("sf",)

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw the scale near the data's mean, its spread, or one.

        The original drew two independent uniforms here, so its three branches
        have probabilities 1/3, 4/9 and 2/9 instead of thirds. This port keeps
        that behaviour.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        if self.sf is not None:
            return
        if rng.random() < 1 / 3:
            loc = math.log(abs(shape["y_mean"]) + 1e-300)
        elif rng.random() < 2 / 3:
            loc = shape["y_sd"]
        else:
            loc = 0.0
        self.sf = rng.normal(loc, sd)

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Return ``sf^2`` everywhere.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            ``[sf]``.
        same : bool
            Ignored.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        ones = torch.ones(x.shape[0], z.shape[0], dtype=torch.float64)
        return torch.exp(2 * theta[0]) * ones


class SqExpKernel(_BaseKernel):
    """Squared exponential (``SE``, GPML ``covSEiso``): a smooth local function."""

    id = "SE"
    param_names = ("lengthscale", "sf")

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw the lengthscale and scale from the data's summary.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        if self.lengthscale is None:
            self.lengthscale = _input_lengthscale(rng, sd, shape, self.dimension)
        if self.sf is None:
            self.sf = _output_scale(rng, sd, shape)

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Return ``sf^2 exp(-r^2 / 2 l^2)``.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            ``[log l, log sf]``.
        same : bool
            Ignored.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        r = _difference(x, z, self.dimension) / torch.exp(theta[0])
        return torch.exp(2 * theta[1]) * torch.exp(-0.5 * r**2)


class GammaExpKernel(_BaseKernel):
    """Gamma-exponential (``GE``): ``sf^2 exp(-(|r| / l)^gamma)``, from AutoGP.

    With ``gamma = 2`` it is a squared exponential of lengthscale
    ``l / sqrt(2)``. Below 2 its sample paths are continuous but nowhere
    differentiable; ``gamma = 1`` gives the Ornstein-Uhlenbeck kernel. It is
    not in the 2013 or 2014 grammar. :mod:`autostat.smc` uses it and the
    report describes it.
    """

    id = "GE"
    param_names = ("lengthscale", "gamma", "sf")

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw the lengthscale and scale as for ``SE``, the exponent in ``(0, 2]``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        if self.lengthscale is None:
            self.lengthscale = _input_lengthscale(rng, sd, shape, self.dimension)
        if self.gamma is None:
            self.gamma = rng.uniform(0.5, 2.0)
        if self.sf is None:
            self.sf = _output_scale(rng, sd, shape)

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Return ``sf^2 exp(-(|r| / l)^gamma)``.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            ``[log l, gamma, log sf]``; the exponent is not on a log scale.
        same : bool
            Ignored.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        r = torch.abs(_difference(x, z, self.dimension)) / torch.exp(theta[0])
        return torch.exp(2 * theta[2]) * torch.exp(-(r ** theta[1]))


class RQKernel(_BaseKernel):
    """Rational quadratic (``RQ``, GPML ``covRQiso``): a scale mixture of SEs."""

    id = "RQ"
    param_names = ("lengthscale", "sf", "alpha")

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw lengthscale, scale and shape.

        The original had no data-driven heuristic for ``alpha`` and drew it
        with twice the usual spread.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        if self.lengthscale is None:
            self.lengthscale = _input_lengthscale(rng, sd, shape, self.dimension)
        if self.sf is None:
            self.sf = _output_scale(rng, sd, shape)
        if self.alpha is None:
            self.alpha = rng.normal(0.0, 2 * sd)

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Return ``sf^2 (1 + r^2 / (2 alpha l^2))^(-alpha)``.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            ``[log l, log sf, log alpha]``.
        same : bool
            Ignored.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        alpha = torch.exp(theta[2])
        r2 = (_difference(x, z, self.dimension) / torch.exp(theta[0])) ** 2
        return torch.exp(2 * theta[1]) * (1 + 0.5 * r2 / alpha) ** (-alpha)


class PeriodicKernel(_BaseKernel):
    """Periodic kernel with its constant removed (``Per``, ``covPeriodicNoDC``).

    ``k(r) = sf^2 (exp(cos(2 pi r / p) / l^2) - I0(1/l^2)) /
    (exp(1/l^2) - I0(1/l^2))``. Subtracting the zero-frequency term makes the
    kernel average to zero over a period, so a periodic component does not
    masquerade as an offset. That is why the authors replaced the textbook
    form, which survives in their code as ``PeriodicKernelOLD``.

    The evaluation goes through the exponentially scaled Bessel function
    ``i0e``: the same quotient multiplied above and below by ``exp(-1/l^2)``.
    It stays stable at every lengthscale for which GPML needed two branches.
    """

    id = "Per"
    param_names = ("lengthscale", "period", "sf")

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw the shape, the period and the scale.

        The three period centres (``0``, ``-3.95`` and ``-5.9`` in log input
        units) are the constants the original ran with. Its data-dependent
        versions are commented out in the 2014 source.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        if self.lengthscale is None:
            self.lengthscale = rng.normal(0.0, sd)
        if self.period is None:
            if rng.random() < 0.33:
                loc = 0.0
            elif rng.random() < 0.5:
                loc = -3.95
            else:
                loc = -5.9
            self.period = rng.normal(loc, sd)
        if self.sf is None:
            self.sf = _output_scale(rng, sd, shape)

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Evaluate the zero-mean periodic covariance.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            ``[log l, log p, log sf]``.
        same : bool
            Ignored.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        ell = torch.exp(theta[0])
        distance = torch.abs(_difference(x, z, self.dimension))
        angle = 2 * math.pi * distance / torch.exp(theta[1])
        inverse = 1 / ell**2
        b0 = torch.special.i0e(inverse)
        closed = (torch.exp((torch.cos(angle) - 1) * inverse) - b0) / (1 - b0)
        profile = torch.where(ell > PERIODIC_ELL_LIMIT, torch.cos(angle), closed)
        return torch.exp(2 * theta[2]) * profile

    def out_of_bounds(self, shape: dict[str, Any]) -> bool:
        """Reject periods shorter than the sampling allows or too long to repeat.

        Parameters
        ----------
        shape : dict
            Data summary, with ``min_period`` and ``max_period``.

        Returns
        -------
        bool
            True outside the bounds.
        """
        d = self.dimension or 0
        min_period = shape.get("min_period")
        too_short = min_period is not None and self.period < min_period[d]
        return too_short or self.period > shape["max_period"][d]


class LinearKernel(_BaseKernel):
    """Linear (``Lin``, GPML ``covLinear``): ``sf^2 (x - location)(z - location)``."""

    id = "Lin"
    param_names = ("sf", "location")

    @property
    def is_stationary(self) -> bool:
        """Report non-stationarity.

        Returns
        -------
        bool
            False.
        """
        return False

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw the slope scale and the location.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        d = self.dimension or 0
        span = shape["x_max"][d] - shape["x_min"][d]
        if self.sf is None:
            draw = rng.random()
            if draw < 1 / 3:
                loc = shape["y_sd"] - shape["x_sd"][d]
            elif draw < 2 / 3:
                loc = math.log(abs((shape["y_max"] - shape["y_min"]) / span))
            else:
                loc = 0.0
            self.sf = rng.normal(loc, sd)
        if self.location is None:
            # Uniform over three times the data range, centred on it.
            lo, hi = shape["x_min"][d], shape["x_max"][d]
            self.location = rng.uniform(2 * lo - hi, 2 * hi - lo)

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Return ``sf^2 (x - c)(z - c)``.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            ``[log sf, c]``.
        same : bool
            Ignored.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        scale = torch.exp(theta[0])
        xs = (_column(x, self.dimension) - theta[1]) * scale
        zs = (_column(z, self.dimension) - theta[1]) * scale
        return xs[:, None] * zs[None, :]


# Operators


class _Operator(Kernel):
    """Shared parameter bookkeeping for kernels built from other kernels."""

    is_operator = True
    arity: ClassVar[int | str] = "n"
    #: Parameters the operator owns, stored before its operands' parameters.
    own_params: ClassVar[tuple[str, ...]] = ()

    @property
    def param_vector(self) -> np.ndarray:
        """Own parameters, then each operand's, in order.

        Returns
        -------
        numpy.ndarray
            The concatenation.
        """
        own = np.array([getattr(self, n) for n in self.own_params], float)
        return np.concatenate([own, *(o.param_vector for o in self.operands)])

    @property
    def num_params(self) -> int:
        """Own parameters plus every operand's.

        Returns
        -------
        int
            The total.
        """
        return len(self.own_params) + sum(o.num_params for o in self.operands)

    def load_param_vector(self, params: Any) -> None:
        """Distribute a flat vector over the operator and its operands.

        Parameters
        ----------
        params : array_like
            Exactly :attr:`num_params` values.
        """
        values = np.asarray(params, float).ravel()
        for i, name in enumerate(self.own_params):
            setattr(self, name, float(values[i]))
        start = len(self.own_params)
        for o in self.operands:
            end = start + o.num_params
            o.load_param_vector(values[start:end])
            start = end

    @property
    def depth(self) -> int:
        """One more than the deepest operand.

        Returns
        -------
        int
            Nesting depth.
        """
        return max(o.depth for o in self.operands) + 1

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Initialise every operand.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        for o in self.operands:
            o.initialise_params(rng, sd, shape)

    def out_of_bounds(self, shape: dict[str, Any]) -> bool:
        """Report whether any operand left its bounds.

        Parameters
        ----------
        shape : dict
            Data summary.

        Returns
        -------
        bool
            True if any operand is out of bounds.
        """
        return any(o.out_of_bounds(shape) for o in self.operands)

    def operand_slices(self, theta: Tensor) -> Iterator[tuple[Kernel, Tensor]]:
        """Pair each operand with its slice of the parameter tensor.

        Parameters
        ----------
        theta : torch.Tensor
            This operator's full parameter tensor.

        Yields
        ------
        tuple of (Kernel, torch.Tensor)
            Operand and its parameters.
        """
        start = len(self.own_params)
        for o in self.operands:
            end = start + o.num_params
            yield o, theta[start:end]
            start = end

    def to_dict(self) -> dict[str, Any]:
        """Describe the operator, its own parameters and its operands.

        Returns
        -------
        dict
            Inverted by :func:`from_dict`.
        """
        out: dict[str, Any] = {"type": type(self).__name__}
        if self.own_params:
            out["dimension"] = self.dimension
        out.update({name: getattr(self, name) for name in self.own_params})
        out["operands"] = [o.to_dict() for o in self.operands]
        return out

    def multiply_by_const(self, sf: float) -> None:
        """Scale every operand.

        Parameters
        ----------
        sf : float
            Log standard deviation of the constant.
        """
        for o in self.operands:
            o.multiply_by_const(sf)


class SumKernel(_Operator):
    """Sum ``k1 + k2 + ...``."""

    id = "Sum"
    is_abelian = True

    def __init__(self, operands: list[Kernel] | None = None) -> None:
        """Create a sum.

        Parameters
        ----------
        operands : list of Kernel, optional
            Summands.
        """
        self.operands = list(operands or [])

    def copy(self) -> Kernel:
        """Return a deep copy.

        Returns
        -------
        Kernel
            New sum of copied operands.
        """
        return SumKernel([o.copy() for o in self.operands])

    @property
    def is_stationary(self) -> bool:
        """Report stationarity when every summand is stationary.

        Returns
        -------
        bool
            The conjunction.
        """
        return all(o.is_stationary for o in self.operands)

    @property
    def sf(self) -> float:
        """Log standard deviation of the sum of stationary summands.

        Returns
        -------
        float
            Half the log of the summed variances.

        Raises
        ------
        RuntimeError
            If a summand is non-stationary.
        """
        if not self.is_stationary:
            raise RuntimeError("non-stationary kernels have no scale factor")
        return 0.5 * math.log(sum(math.exp(2 * o.sf) for o in self.operands))

    @property
    def effective_params(self) -> int:
        """Total of the summands' effective parameters.

        Returns
        -------
        int
            The sum.
        """
        return sum(o.effective_params for o in self.operands)

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Add the operands' covariances.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            Operands' parameters, concatenated.
        same : bool
            Passed to the operands.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        out = torch.zeros(x.shape[0], z.shape[0], dtype=torch.float64)
        for o, t in self.operand_slices(theta):
            out = out + o.cov(x, z, t, same)
        return out

    def __repr__(self) -> str:
        """Render the original's format.

        Returns
        -------
        str
            ``SumKernel(operands=[...])``.
        """
        return f"SumKernel(operands=[{', '.join(repr(o) for o in self.operands)}])"

    def pretty_print(self) -> str:
        """Render the summands joined by ``+``.

        Returns
        -------
        str
            Parenthesised sum.
        """
        return "( " + " + ".join(o.pretty_print() for o in self.operands) + " )"

    def structure(self) -> str:
        """Render the summands' structures joined by ``+``.

        Returns
        -------
        str
            E.g. ``WN + SE``.
        """
        return " + ".join(o.structure() for o in self.operands)


class ProductKernel(_Operator):
    """Product ``k1 x k2 x ...``."""

    id = "Product"
    is_abelian = True

    def __init__(self, operands: list[Kernel] | None = None) -> None:
        """Create a product.

        Parameters
        ----------
        operands : list of Kernel, optional
            Factors.
        """
        self.operands = list(operands or [])

    def copy(self) -> Kernel:
        """Return a deep copy.

        Returns
        -------
        Kernel
            New product of copied operands.
        """
        return ProductKernel([o.copy() for o in self.operands])

    @property
    def is_stationary(self) -> bool:
        """Report stationarity when every factor is stationary.

        Returns
        -------
        bool
            The conjunction.
        """
        return all(o.is_stationary for o in self.operands)

    @property
    def sf(self) -> float:
        """Log standard deviation: the factors' log scales add up.

        Returns
        -------
        float
            The sum of log scales.

        Raises
        ------
        RuntimeError
            If a factor is non-stationary.
        """
        if not self.is_stationary:
            raise RuntimeError("non-stationary kernels have no scale factor")
        return sum(o.sf for o in self.operands)

    @property
    def effective_params(self) -> int:
        """Factors' parameters minus the redundant scale of each extra factor.

        Returns
        -------
        int
            The count BIC uses.
        """
        total = sum(o.effective_params for o in self.operands)
        return total - (len(self.operands) - 1)

    def multiply_by_const(self, sf: float) -> None:
        """Scale the first factor only, which scales the product once.

        Parameters
        ----------
        sf : float
            Log standard deviation of the constant.
        """
        self.operands[0].multiply_by_const(sf)

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Multiply the operands' covariances elementwise.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            Operands' parameters, concatenated.
        same : bool
            Passed to the operands.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        out = torch.ones(x.shape[0], z.shape[0], dtype=torch.float64)
        for o, t in self.operand_slices(theta):
            out = out * o.cov(x, z, t, same)
        return out

    def __repr__(self) -> str:
        """Render the original's format.

        Returns
        -------
        str
            ``ProductKernel(operands=[...])``.
        """
        return f"ProductKernel(operands=[{', '.join(repr(o) for o in self.operands)}])"

    def pretty_print(self) -> str:
        """Render the factors joined by ``x``.

        Returns
        -------
        str
            Parenthesised product.
        """
        return "( " + " x ".join(o.pretty_print() for o in self.operands) + " )"

    def structure(self) -> str:
        """Render the factors' structures joined by ``x``.

        Returns
        -------
        str
            E.g. ``Per x SE``; sums inside are parenthesised.
        """
        parts = [
            f"({o.structure()})" if isinstance(o, SumKernel) else o.structure()
            for o in self.operands
        ]
        return " x ".join(parts)


class _Switch(_Operator):
    """Shared code for the two operators that switch between two kernels."""

    arity = 2
    is_abelian = False

    @property
    def is_stationary(self) -> bool:
        """Report non-stationarity: the switch depends on absolute position.

        Returns
        -------
        bool
            False.
        """
        return False

    def weight(self, x: Tensor, theta: Tensor) -> Tensor:
        """Return the weight of the second operand at each input.

        Parameters
        ----------
        x : torch.Tensor
            Inputs.
        theta : torch.Tensor
            The operator's own parameters first.

        Returns
        -------
        torch.Tensor
            Values in ``[0, 1]``, shape ``(n,)``.

        Raises
        ------
        NotImplementedError
            Always: the change-point and change-window operators override it.
        """
        raise NotImplementedError

    def cov(self, x: Tensor, z: Tensor, theta: Tensor, same: bool) -> Tensor:
        """Blend: ``(1 - w(x)) k1 (1 - w(z)) + w(x) k2 w(z)``.

        Parameters
        ----------
        x, z : torch.Tensor
            Inputs.
        theta : torch.Tensor
            Own parameters, then the two operands'.
        same : bool
            Passed to the operands.

        Returns
        -------
        torch.Tensor
            Shape ``(n, m)``.
        """
        wx, wz = self.weight(x, theta), self.weight(z, theta)
        (k1, t1), (k2, t2) = self.operand_slices(theta)
        first = (1 - wx)[:, None] * k1.cov(x, z, t1, same) * (1 - wz)[None, :]
        second = wx[:, None] * k2.cov(x, z, t2, same) * wz[None, :]
        return first + second

    def __repr__(self) -> str:
        """Render the original's format.

        Returns
        -------
        str
            Constructor call with own parameters then operands.
        """
        own = ", ".join(f"{n}={getattr(self, n)!r}" for n in self.own_params)
        ops = ", ".join(repr(o) for o in self.operands)
        head = f"{type(self).__name__}(dimension={self.dimension}, {own}"
        return f"{head}, operands=[ {ops} ])"

    def pretty_print(self) -> str:
        """Render own parameters to one decimal, then both operands.

        Returns
        -------
        str
            E.g. ``CP(dim=0, location=1974.1, steepness=3.8, ..., ...)``.
        """
        own = ", ".join(f"{n}={_fmt(getattr(self, n))}" for n in self.own_params)
        a, b = (o.pretty_print() for o in self.operands)
        return f"{self.id}(dim={self.dimension}, {own}, {a}, {b})"

    def structure(self) -> str:
        """Render ``CP(k1, k2)`` or ``CW(k1, k2)``.

        Returns
        -------
        str
            The structure.
        """
        return f"{self.id}({', '.join(o.structure() for o in self.operands)})"


class ChangePointKernel(_Switch):
    """Changepoint (``CP``, ``covChangePointMultiD``): a sigmoidal switch.

    The weight of the second kernel is
    ``(1 + tanh((x - location) exp(steepness))) / 2``.
    """

    id = "CP"
    own_params = ("location", "steepness")

    def __init__(
        self,
        dimension: int | None = None,
        location: float | None = None,
        steepness: float | None = None,
        operands: list[Kernel] | None = None,
    ) -> None:
        """Create a changepoint between two kernels.

        Parameters
        ----------
        dimension : int, optional
            Input dimension of the switch.
        location : float, optional
            Where the switch is half-way.
        steepness : float, optional
            Log of the switch's inverse width.
        operands : list of Kernel
            Exactly two kernels: before, then after.

        Raises
        ------
        ValueError
            Unless there are exactly two operands.
        """
        self.dimension = dimension
        self.location = location
        self.steepness = steepness
        self.operands = list(operands or [])
        if len(self.operands) != 2:
            raise ValueError("a changepoint switches between exactly two kernels")

    def copy(self) -> Kernel:
        """Return a deep copy.

        Returns
        -------
        Kernel
            Same parameters, copied operands.
        """
        return ChangePointKernel(
            self.dimension,
            self.location,
            self.steepness,
            [o.copy() for o in self.operands],
        )

    @property
    def effective_params(self) -> int:
        """Location and steepness plus the operands'.

        Returns
        -------
        int
            The count BIC uses.
        """
        return 2 + sum(o.effective_params for o in self.operands)

    def weight(self, x: Tensor, theta: Tensor) -> Tensor:
        """Sigmoid rising through ``location``.

        Parameters
        ----------
        x : torch.Tensor
            Inputs.
        theta : torch.Tensor
            ``[location, log steepness, ...]``.

        Returns
        -------
        torch.Tensor
            Shape ``(n,)``.
        """
        u = (_column(x, self.dimension) - theta[0]) * torch.exp(theta[1])
        return 0.5 + 0.5 * torch.tanh(u)

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw a location inside the data and a steepness from its span.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        d = self.dimension or 0
        span = shape["x_max"][d] - shape["x_min"][d]
        if self.location is None:
            self.location = rng.uniform(shape["x_min"][d], shape["x_max"][d])
        if self.steepness is None:
            self.steepness = rng.normal(3.3 - math.log(span), 1.0)
        super().initialise_params(rng, sd, shape)

    def out_of_bounds(self, shape: dict[str, Any]) -> bool:
        """Reject changepoints outside the data or too gradual to mean anything.

        Parameters
        ----------
        shape : dict
            Data summary.

        Returns
        -------
        bool
            True outside the bounds.
        """
        d = self.dimension or 0
        span = shape["x_max"][d] - shape["x_min"][d]
        return (
            self.location < shape["x_min"][d]
            or self.location > shape["x_max"][d]
            or self.steepness < -math.log(span) + 2.3
            or super().out_of_bounds(shape)
        )


class ChangeWindowKernel(_Switch):
    """Change window (``CW``, ``covChangeWindowMultiD``).

    The second kernel acts inside a window: the product of a rising and a
    falling sigmoid, centred on ``location``, of width ``exp(width)``.
    """

    id = "CW"
    own_params = ("location", "steepness", "width")

    def __init__(
        self,
        dimension: int | None = None,
        location: float | None = None,
        steepness: float | None = None,
        width: float | None = None,
        operands: list[Kernel] | None = None,
    ) -> None:
        """Create a change window between two kernels.

        Parameters
        ----------
        dimension : int, optional
            Input dimension of the window.
        location : float, optional
            Centre of the window.
        steepness : float, optional
            Log of the edges' inverse width.
        width : float, optional
            Log of the window's width.
        operands : list of Kernel
            Exactly two kernels: outside, then inside.

        Raises
        ------
        ValueError
            Unless there are exactly two operands.
        """
        self.dimension = dimension
        self.location = location
        self.steepness = steepness
        self.width = width
        self.operands = list(operands or [])
        if len(self.operands) != 2:
            raise ValueError("a change window switches between exactly two kernels")

    def copy(self) -> Kernel:
        """Return a deep copy.

        Returns
        -------
        Kernel
            Same parameters, copied operands.
        """
        return ChangeWindowKernel(
            self.dimension,
            self.location,
            self.steepness,
            self.width,
            [o.copy() for o in self.operands],
        )

    @property
    def effective_params(self) -> int:
        """Location, steepness and width plus the operands'.

        Returns
        -------
        int
            The count BIC uses.
        """
        return 3 + sum(o.effective_params for o in self.operands)

    def weight(self, x: Tensor, theta: Tensor) -> Tensor:
        """Smooth indicator of the window.

        Parameters
        ----------
        x : torch.Tensor
            Inputs.
        theta : torch.Tensor
            ``[location, log steepness, log width, ...]``.

        Returns
        -------
        torch.Tensor
            Shape ``(n,)``.
        """
        steep = torch.exp(theta[1])
        half = 0.5 * torch.exp(theta[2])
        col = _column(x, self.dimension)
        rise = 0.5 + 0.5 * torch.tanh((col - (theta[0] - half)) * steep)
        fall = 0.5 + 0.5 * torch.tanh(-(col - (theta[0] + half)) * steep)
        return rise * fall

    def initialise_params(
        self, rng: np.random.Generator, sd: float, shape: dict[str, Any]
    ) -> None:
        """Draw a location inside the data, a steepness and a small width.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness.
        sd : float
            Spread on the log scale.
        shape : dict
            Data summary.
        """
        d = self.dimension or 0
        span = shape["x_max"][d] - shape["x_min"][d]
        if self.location is None:
            self.location = rng.uniform(shape["x_min"][d], shape["x_max"][d])
        if self.steepness is None:
            self.steepness = rng.normal(3.3 - math.log(span), 1.0)
        if self.width is None:
            self.width = rng.normal(math.log(0.1 * span), 1.0)
        super().initialise_params(rng, sd, shape)

    def out_of_bounds(self, shape: dict[str, Any]) -> bool:
        """Reject windows touching the data's edges, too wide or too gradual.

        Parameters
        ----------
        shape : dict
            Data summary.

        Returns
        -------
        bool
            True outside the bounds.
        """
        d = self.dimension or 0
        lo, hi = shape["x_min"][d], shape["x_max"][d]
        span = hi - lo
        half = math.exp(self.width) / 2
        return (
            self.location - half < lo + 0.05 * span
            or self.location + half > hi - 0.05 * span
            or self.width > math.log(0.25 * span)
            or self.steepness < -math.log(span) + 2.3
            or super().out_of_bounds(shape)
        )


# Registry and serialisation

#: Base kernels the grammar introduces, keyed by the identifiers the original
#: used in its ``base_kernels`` strings. ``SP``, ``PerOLD`` and ``LinOLD`` were
#: experimental variants absent from every published configuration.
BASE_KERNELS: dict[str, type[Kernel]] = {
    "SE": SqExpKernel,
    "Const": ConstKernel,
    "Lin": LinearKernel,
    "Per": PeriodicKernel,
    "RQ": RQKernel,
    "Noise": NoiseKernel,
    "GE": GammaExpKernel,
}

KERNEL_CLASSES: dict[str, type[Kernel]] = {
    cls.__name__: cls
    for cls in (
        ZeroKernel,
        NoneKernel,
        NoiseKernel,
        ConstKernel,
        SqExpKernel,
        GammaExpKernel,
        RQKernel,
        PeriodicKernel,
        LinearKernel,
        SumKernel,
        ProductKernel,
        ChangePointKernel,
        ChangeWindowKernel,
    )
}


def base_kernels(dimensions: int, names: str) -> Iterator[Kernel]:
    """Yield every base kernel named in ``names``, once per input dimension.

    Parameters
    ----------
    dimensions : int
        Number of input dimensions.
    names : str
        Comma-separated identifiers, e.g. ``"SE,Per,Lin,Const,Noise"``.

    Yields
    ------
    Kernel
        Unparameterised base kernels.

    Raises
    ------
    ValueError
        On an unknown identifier.
    """
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    unknown = sorted(set(wanted) - set(BASE_KERNELS))
    if unknown:
        known = sorted(BASE_KERNELS)
        raise ValueError(f"unknown base kernels {unknown}; known: {known}")
    for kid, cls in BASE_KERNELS.items():
        if kid not in wanted:
            continue
        if cls.is_thunk:
            yield cls()
        else:
            for d in range(dimensions):
                yield cls(dimension=d)


def from_dict(spec: dict[str, Any]) -> Kernel:
    """Rebuild a kernel from :meth:`Kernel.to_dict` output.

    Parameters
    ----------
    spec : dict
        As produced by ``to_dict``.

    Returns
    -------
    Kernel
        Equal to the serialised kernel.

    Raises
    ------
    ValueError
        On an unknown kernel type.
    """
    spec = dict(spec)
    name = spec.pop("type")
    if name not in KERNEL_CLASSES:
        raise ValueError(f"unknown kernel type {name!r}")
    cls = KERNEL_CLASSES[name]
    if "operands" in spec:
        spec["operands"] = [from_dict(o) for o in spec["operands"]]
    if cls in (SumKernel, ProductKernel):
        return cls(spec["operands"])
    return cls(**spec)


def remove_duplicates(things: list[Any]) -> list[Any]:
    """Keep the first of each group of objects with equal representations.

    The original used ``list(set(things))``, whose order depends on hash
    randomisation. Its searches were therefore not reproducible from one run
    to the next. First-seen order fixes that.

    Parameters
    ----------
    things : list
        Kernels or models.

    Returns
    -------
    list
        Unique by ``repr``, in first-seen order.
    """
    seen: dict[str, Any] = {}
    for thing in things:
        seen.setdefault(repr(thing), thing)
    return list(seen.values())
