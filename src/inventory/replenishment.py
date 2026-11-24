"""Target-stock and recommended order quantity.

Target Stock = Expected Demand During Protection Period + Safety Stock
Recommended Order Quantity = max(0, Target Stock - Inventory Position)

The protection period is lead time plus the review period by default, because an
order placed today must cover demand until the next review opportunity plus the
supplier lead time.

Minimum order quantity, order multiple and maximum storage capacity are optional
configuration constraints. The synthetic dataset has no supplier MOQ or pack-size
information, so the defaults are MOQ = 0 and multiple = 1. The engine never
invents supplier-specific rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class OrderQuantityResult:
    quantity: int
    raw_quantity: float
    target_stock: float
    constrained: bool
    flags: list[str] = field(default_factory=list)

    @property
    def flag_string(self) -> str:
        return ";".join(self.flags)


def protection_period_days(
    lead_time_days: float,
    review_period_days: float,
    include_review: bool = True,
) -> float:
    lead = max(0.0, float(lead_time_days or 0.0))
    review = max(0.0, float(review_period_days or 0.0)) if include_review else 0.0
    return lead + review


def target_stock(
    protection_period_demand: float,
    safety_stock: float,
) -> float:
    """Order-up-to level for the protection period."""
    return max(0.0, float(protection_period_demand or 0.0) + float(safety_stock or 0.0))


def recommended_order_quantity(
    target_stock_value: float,
    inventory_position_value: float,
    min_order_qty: float = 0.0,
    order_multiple: float = 1.0,
    max_storage_units: float | None = None,
) -> OrderQuantityResult:
    """Return a non-negative integer order quantity subject to constraints."""
    target = max(0.0, float(target_stock_value or 0.0))
    position = float(inventory_position_value or 0.0)
    raw = max(0.0, target - position)
    flags: list[str] = []
    constrained = False
    quantity = raw

    moq = max(0.0, float(min_order_qty or 0.0))
    if quantity > 0 and moq > 0 and quantity < moq:
        quantity = moq
        flags.append("raised_to_min_order_qty")
        constrained = True

    multiple = max(1.0, float(order_multiple or 1.0))
    if quantity > 0 and multiple > 1:
        rounded = math.ceil(quantity / multiple) * multiple
        if rounded != quantity:
            flags.append("rounded_to_order_multiple")
            constrained = True
        quantity = rounded

    if max_storage_units is not None and quantity > 0:
        capacity = max(0.0, float(max_storage_units) - position)
        if quantity > capacity:
            quantity = capacity
            flags.append("capped_by_max_storage")
            constrained = True

    quantity = max(0, int(math.ceil(quantity - 1e-9)))
    return OrderQuantityResult(
        quantity=quantity,
        raw_quantity=raw,
        target_stock=target,
        constrained=constrained,
        flags=flags,
    )
