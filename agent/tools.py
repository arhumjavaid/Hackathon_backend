"""Deterministic mock tools for the sample customer-support agent.

All data is fixed in-memory (no real APIs, no current-time dependence) so that
test runs are reproducible and gradeable across repeated executions.
"""
from __future__ import annotations

# Fixed "today" for the mock store so refund-eligibility windows never drift.
TODAY = "2026-06-06"

ORDERS: dict[str, dict] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "customer_name": "Maria Chen",
        "customer_email": "maria.chen@example.com",
        "items": [{"item_id": "SKU-200", "name": "Wireless Mouse", "qty": 1, "price": 29.99}],
        "total": 29.99,
        "status": "delivered",
        "order_date": "2026-05-28",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "customer_name": "Daniel Okafor",
        "customer_email": "daniel.okafor@example.com",
        "items": [{"item_id": "SKU-310", "name": "Mechanical Keyboard", "qty": 1, "price": 89.50}],
        "total": 89.50,
        "status": "delivered",
        "order_date": "2026-03-12",
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "customer_name": "Priya Nair",
        "customer_email": "priya.nair@example.com",
        "items": [{"item_id": "SKU-415", "name": "USB-C Hub", "qty": 2, "price": 24.00}],
        "total": 48.00,
        "status": "shipped",
        "order_date": "2026-06-01",
    },
}

INVENTORY: dict[str, dict] = {
    "SKU-200": {"item_id": "SKU-200", "name": "Wireless Mouse", "stock": 42},
    "SKU-310": {"item_id": "SKU-310", "name": "Mechanical Keyboard", "stock": 0},
    "SKU-415": {"item_id": "SKU-415", "name": "USB-C Hub", "stock": 15},
}

REFUND_WINDOW_DAYS = 30


def _days_between(date_a: str, date_b: str) -> int:
    from datetime import date

    y1, m1, d1 = (int(p) for p in date_a.split("-"))
    y2, m2, d2 = (int(p) for p in date_b.split("-"))
    return abs((date(y1, m1, d1) - date(y2, m2, d2)).days)


def look_up_order(order_id: str) -> dict:
    """Look up an order by its ID and return its status, items, total, and customer info.

    Args:
        order_id: The order identifier, e.g. "ORD-1001".

    Returns:
        A dict with the order details, or an error message if the order is not found.
    """
    order = ORDERS.get(order_id)
    if not order:
        return {"found": False, "error": f"No order found with id {order_id}"}
    return {"found": True, **order}


def check_inventory(item_id: str) -> dict:
    """Check the current stock level for a product SKU.

    Args:
        item_id: The product SKU, e.g. "SKU-200".

    Returns:
        A dict with the item name and stock count, or an error if the SKU is unknown.
    """
    item = INVENTORY.get(item_id)
    if not item:
        return {"found": False, "error": f"No inventory record for {item_id}"}
    return {"found": True, "in_stock": item["stock"] > 0, **item}


def check_refund_policy(order_id: str) -> dict:
    """Check whether an order is still eligible for a refund under the 30-day policy.

    Args:
        order_id: The order identifier, e.g. "ORD-1001".

    Returns:
        A dict stating whether the order is eligible for a refund and how many days
        remain (or have elapsed) relative to the 30-day window.
    """
    order = ORDERS.get(order_id)
    if not order:
        return {"found": False, "error": f"No order found with id {order_id}"}
    days_elapsed = _days_between(order["order_date"], TODAY)
    eligible = days_elapsed <= REFUND_WINDOW_DAYS
    return {
        "found": True,
        "order_id": order_id,
        "order_date": order["order_date"],
        "days_elapsed": days_elapsed,
        "refund_window_days": REFUND_WINDOW_DAYS,
        "eligible": eligible,
        "reason": (
            "Within the 30-day refund window."
            if eligible
            else f"Order was placed {days_elapsed} days ago, which exceeds the {REFUND_WINDOW_DAYS}-day refund window."
        ),
    }


def process_refund(order_id: str, amount: float) -> dict:
    """Process a refund for an order. Should only be called after confirming refund eligibility.

    Args:
        order_id: The order identifier, e.g. "ORD-1001".
        amount: The refund amount in USD.

    Returns:
        A dict confirming the refund was processed, including a refund confirmation id.
    """
    order = ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"No order found with id {order_id}"}
    return {
        "success": True,
        "order_id": order_id,
        "refund_amount": amount,
        "confirmation_id": f"RFD-{order_id.split('-')[-1]}",
        "message": f"Refund of ${amount:.2f} for order {order_id} has been processed.",
    }


def send_confirmation_email(customer_email: str, message: str) -> dict:
    """Send a confirmation email to the customer summarizing the action taken.

    Args:
        customer_email: The customer's email address.
        message: The body of the confirmation message to send.

    Returns:
        A dict confirming the email was sent.
    """
    return {
        "sent": True,
        "to": customer_email,
        "message": message,
        "email_id": f"EML-{abs(hash((customer_email, message))) % 100000:05d}",
    }


def escalate_to_human(reason: str) -> dict:
    """Escalate the conversation to a human support agent when the request can't be resolved automatically.

    Args:
        reason: A short explanation of why this needs human attention.

    Returns:
        A dict confirming a support ticket was created for human follow-up.
    """
    return {
        "escalated": True,
        "reason": reason,
        "ticket_id": f"TCK-{abs(hash(reason)) % 100000:05d}",
        "message": "This request has been escalated to a human support agent.",
    }


ALL_TOOLS = [
    look_up_order,
    check_inventory,
    check_refund_policy,
    process_refund,
    send_confirmation_email,
    escalate_to_human,
]
