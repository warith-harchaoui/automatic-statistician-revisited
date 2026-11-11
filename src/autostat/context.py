"""What the business already knows about a series, stated once and checked.

What this module does
    A :class:`Context` writes down what a domain expert knows before looking
    at the data: the unit of the values, whether they are positive or
    multiplicative, the cycles one expects (a week, a year) and the dated
    events that may have moved the series (a holiday, a promotion, a change
    of rule). It reads from a YAML file or a dictionary, so the same context
    serves a whole family of series.

Why it exists
    The structure search is blind to meaning: for it, sales and heartbeats
    are numbers. Features measured against what the business expects say
    more than raw parameters: "the expected weekly cycle is absent" is a
    finding, "a periodic kernel of period 0.0192" is a number to decode.
    The context never decides the model: the search and its BIC do. It only
    names what to look for and what to compare with.

What it consumes
    A mapping such as::

        name: retail sales
        value_unit: units
        positive: true
        cycles:
          - {name: yearly, period: 1y}
          - {name: weekly, period: 7d, tolerance: 0.03}
        events:
          - {name: covid lockdown, at: 2020-03-17}

    Periods take a suffix (``h``, ``d``, ``w``, ``mo``, ``y``) or a bare
    number in the series' time unit; dates are ISO dates or bare numbers.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Length of each period suffix, in years.
YEARS = {"h": 1 / 8766, "d": 1 / 365.25, "w": 7 / 365.25, "mo": 1 / 12, "y": 1.0}


def years(value: float | str) -> float:
    """Read a duration as a number of years.

    Parameters
    ----------
    value : float or str
        A bare number (already in years) or a number with a suffix of
        :data:`YEARS`, e.g. ``"7d"`` or ``"12mo"``.

    Returns
    -------
    float
        The duration in years.
    """
    if isinstance(value, int | float):
        return float(value)
    match = re.fullmatch(r"\s*([0-9.]+)\s*(h|d|w|mo|y)\s*", str(value))
    if not match:
        raise ValueError(f"unreadable duration {value!r}: use e.g. 7d, 12mo or 1y")
    return float(match.group(1)) * YEARS[match.group(2)]


def decimal_year(value: float | str | dt.date) -> float:
    """Read a date as a decimal year, the time axis of the reports.

    Parameters
    ----------
    value : float, str or datetime.date
        A bare number (already a decimal year), an ISO date or a date.

    Returns
    -------
    float
        E.g. ``2020.208`` for 17 March 2020.
    """
    if isinstance(value, int | float):
        return float(value)
    day = value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
    start = dt.date(day.year, 1, 1)
    length = (dt.date(day.year + 1, 1, 1) - start).days
    return day.year + (day - start).days / length


@dataclass
class Cycle:
    """A cycle the business expects.

    Parameters
    ----------
    name : str
        How the business calls it, e.g. ``"weekly"``.
    period : float
        Its period, in years.
    tolerance : float
        Relative gap within which a fitted period counts as this cycle.
    """

    name: str
    period: float
    tolerance: float = 0.05


@dataclass
class Event:
    """A dated event that may have moved the series.

    Parameters
    ----------
    name : str
        How the business calls it.
    at : float
        Its date, as a decimal year.
    tolerance : float
        Gap in years within which a fitted changepoint counts as this event.
    """

    name: str
    at: float
    tolerance: float = 0.25


@dataclass
class Context:
    """What the business knows about a family of series.

    Parameters
    ----------
    name : str
        The family, e.g. ``"retail sales"``.
    value_unit : str
        Unit of the values, for the feature names' readers.
    positive : bool
        The values cannot be negative (counts, prices, sales).
    multiplicative : bool
        Cycles scale with the level, so their size is read in per cent of it.
    cycles : list of Cycle
        Expected cycles.
    events : list of Event
        Dated events.
    """

    name: str = ""
    value_unit: str = ""
    positive: bool = False
    multiplicative: bool = False
    cycles: list[Cycle] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Context:
        """Build a context from a mapping, as read from YAML.

        Parameters
        ----------
        data : dict
            Keys of :class:`Context`; cycles and events as lists of mappings.

        Returns
        -------
        Context
            The context, with durations in years and dates as decimal years.
        """
        cycles = [
            Cycle(c["name"], years(c["period"]), float(c.get("tolerance", 0.05)))
            for c in data.get("cycles", [])
        ]
        events = [
            Event(e["name"], decimal_year(e["at"]), years(e.get("tolerance", 0.25)))
            for e in data.get("events", [])
        ]
        return cls(
            name=str(data.get("name", "")),
            value_unit=str(data.get("value_unit", "")),
            positive=bool(data.get("positive", False)),
            multiplicative=bool(data.get("multiplicative", False)),
            cycles=cycles,
            events=events,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Context:
        """Read a context from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file.

        Returns
        -------
        Context
            As :meth:`from_dict`.
        """
        import yaml

        return cls.from_dict(
            yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        )
