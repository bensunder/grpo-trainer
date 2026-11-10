"""Deterministic synthetic supplier invoices for the extraction proof point (no customer data needed).

Plain-text invoices in four layouts with the traps that make real invoice extraction hard:
  * the grand total sits next to "Amount paid" and "Balance due" (partial payments) and a shipping line;
  * dates in seven formats; numeric day/month formats only when unambiguous (day > 12);
  * EUR amounts sometimes in European notation (1.234,56);
  * PO / reference numbers under varying labels, and often absent (tempting hallucination);
  * tax sometimes absent (exempt), due date sometimes absent;
  * distractor numbers: phone, tax ids, bank account, page numbers.

Labels are exact by construction (amounts computed in Decimal, ROUND_HALF_UP). Same seed -> same file.
This is a capability demonstration, not a substitute for the customer's own documents.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

CENT = Decimal("0.01")

VENDORS = [
    "Northwind Industrial Supply",
    "Apex Fabrication Co.",
    "Bluebird Logistics LLC",
    "Cedar Ridge Analytics",
    "Halcyon Medical Devices",
    "Ironclad Security Services",
    "Juniper Office Solutions",
    "Keystone Electrical",
    "Lumen Cloud Hosting",
    "Meridian Freight GmbH",
    "Nordic Timber AB",
    "Orion Lab Instruments",
    "Pinnacle Staffing Group",
    "Quarry Stone & Tile",
    "Redwood Legal Partners",
    "Summit Outdoor Gear",
    "Tidewater Marine Parts",
    "Uplink Telecom Ltd",
    "Vantage Print Works",
    "Westbrook Catering",
]
CUSTOMERS = [
    "Acme Corporation",
    "Brightline Health",
    "Crescent Energy Partners",
    "Delta Water Utility",
    "Evergreen School District",
    "Foxhollow Manufacturing",
    "Granite Peak Construction",
    "Harbor City Hospital",
    "Ironwood Data Centers",
    "Jasper County Transit",
    "Kestrel Aerospace",
    "Lakeside Property Group",
]
ITEMS = [
    ("Hydraulic hose assembly 3/8 in", 18, 140),
    ("Stainless fasteners, box of 100", 9, 45),
    ("Safety gloves (pair)", 3, 12),
    ("Consulting services (hours)", 95, 240),
    ("Cloud compute, monthly", 120, 900),
    ("Forklift rental (days)", 160, 420),
    ("Printed brochures, 500 ct", 85, 310),
    ("Network switch 24-port", 180, 620),
    ("On-site security guard (hours)", 28, 55),
    ("Laboratory pipette set", 60, 210),
    ("Catering, per head", 18, 65),
    ("Freight, LTL pallet", 90, 380),
    ("Copper wire 12 AWG, 250 ft", 70, 160),
    ("Legal review (hours)", 220, 650),
    ("Ceramic floor tile, sq ft", 2, 9),
    ("Marine bilge pump", 140, 520),
    ("Annual software license", 400, 5200),
    ("Ergonomic office chair", 190, 780),
    ("Temporary staffing (hours)", 32, 85),
    ("Replacement filter cartridge", 14, 60),
]
CURRENCIES = {
    "USD": {"tax": ("Sales Tax", [Decimal("0.0625"), Decimal("0.0725"), Decimal("0.0825")])},
    "EUR": {"tax": ("VAT", [Decimal("0.19"), Decimal("0.20"), Decimal("0.21")])},
    "GBP": {"tax": ("VAT", [Decimal("0.20")])},
    "CAD": {"tax": ("GST/HST", [Decimal("0.05"), Decimal("0.13")])},
}
INVOICE_LABELS = ["Invoice #", "Invoice No.", "Invoice Number", "INV No", "Bill No."]
PO_LABELS = ["PO #", "Purchase Order", "Your Ref", "Customer PO", "P.O. Number"]
DATE_LABELS = ["Invoice Date", "Date", "Issued", "Date of Issue"]
DUE_LABELS = ["Due Date", "Payment Due", "Pay By"]
STREETS = ["Industrial Pkwy", "Market St", "Harbor Rd", "Elm Ave"]


@dataclass(frozen=True)
class Invoice:
    text: str
    label: dict[str, Any]


def _money(rng: random.Random, amount: Decimal, currency: str) -> str:
    q = amount.quantize(CENT, rounding=ROUND_HALF_UP)
    us = f"{q:,.2f}"
    if currency == "EUR" and rng.random() < 0.5:
        eu = us.replace(",", "_").replace(".", ",").replace("_", ".")
        return rng.choice([f"€{eu}", f"{eu} EUR", f"EUR {eu}"])
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$"}[currency]
    style = rng.random()
    if style < 0.5:
        return f"{sym}{us}"
    if style < 0.8:
        return f"{currency} {us}"
    return f"{us} {currency}" if currency != "USD" or rng.random() < 0.5 else f"{sym}{us}"


def _fmt_date(rng: random.Random, d: date) -> str:
    options = ["iso", "long_us", "long_eu", "abbr"]
    if d.day > 12:
        options += ["us_num", "eu_num", "dot"]
    kind = rng.choice(options)
    if kind == "iso":
        return d.isoformat()
    if kind == "long_us":
        return f"{d:%B} {d.day}, {d.year}"
    if kind == "long_eu":
        return f"{d.day} {d:%B} {d.year}"
    if kind == "abbr":
        return f"{d.day:02d}-{d:%b}-{d.year}"
    if kind == "us_num":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if kind == "eu_num":
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    return f"{d.day:02d}.{d.month:02d}.{d.year}"


def _phone(rng: random.Random) -> str:
    return f"({rng.randint(200, 989)}) {rng.randint(200, 989)}-{rng.randint(1000, 9999)}"


def make_invoice(rng: random.Random, serial: str) -> Invoice:
    currency = rng.choices(list(CURRENCIES), weights=[5, 2, 1.5, 1.5])[0]
    vendor = rng.choice(VENDORS)
    customer = rng.choice(CUSTOMERS)
    inv_no = rng.choice(
        [f"INV-{serial}", f"{rng.randint(10, 99)}-{serial}", f"{serial}{rng.choice('ABCDE')}"]
    )
    issued = date(2024, 1, 1) + timedelta(days=rng.randint(0, 700))
    due = issued + timedelta(days=rng.choice([15, 30, 45, 60])) if rng.random() < 0.75 else None
    po = f"{rng.choice(['PO', 'P', '4500'])}{rng.randint(10000, 99999)}" if rng.random() < 0.6 else None

    lines = []
    for _ in range(rng.randint(1, 5)):
        desc, lo, hi = rng.choice(ITEMS)
        unit = Decimal(rng.randint(lo * 100, hi * 100)) / 100
        hourly = "(hours)" in desc
        qty = (
            Decimal(rng.choice([1, 1.5, 2, 3.5, 4, 6, 7.5, 8, 12])) if hourly else Decimal(rng.randint(1, 40))
        )
        amount = (qty * unit).quantize(CENT, rounding=ROUND_HALF_UP)
        lines.append({"description": desc, "quantity": qty, "unit_price": unit, "amount": amount})
    subtotal = sum((li["amount"] for li in lines), Decimal(0))
    tax_name, rates = CURRENCIES[currency]["tax"]
    exempt = rng.random() < 0.2
    rate = rng.choice(rates)
    tax = None if exempt else (subtotal * rate).quantize(CENT, rounding=ROUND_HALF_UP)
    shipping = Decimal(rng.randint(15, 180)) if rng.random() < 0.35 else Decimal(0)
    total = subtotal + (tax or 0) + shipping
    paid = (total * Decimal(rng.choice(["0.25", "0.5"]))).quantize(CENT) if rng.random() < 0.3 else Decimal(0)

    m = lambda a: _money(rng, a, currency)  # noqa: E731 - local formatter
    hdr = [
        f"{rng.choice(INVOICE_LABELS)}: {inv_no}",
        f"{rng.choice(DATE_LABELS)}: {_fmt_date(rng, issued)}",
    ]
    if due:
        hdr.append(f"{rng.choice(DUE_LABELS)}: {_fmt_date(rng, due)}")
    if po:
        hdr.append(f"{rng.choice(PO_LABELS)}: {po}")
    rng.shuffle(hdr)

    vendor_block = [
        vendor,
        f"{rng.randint(10, 9999)} {rng.choice(STREETS)}, Suite {rng.randint(1, 900)}",
        f"Tel {_phone(rng)}  |  Tax ID {rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
    ]
    bill_to = [f"Bill To: {customer}", f"Attn: Accounts Payable, Dept {rng.randint(100, 999)}"]

    totals = [f"Subtotal: {m(subtotal)}"]
    if shipping:
        totals.append(f"Shipping & Handling: {m(shipping)}")
    totals.append(
        f"{tax_name} (exempt): {m(Decimal(0))}" if exempt else f"{tax_name} ({rate * 100:.2f}%): {m(tax)}"
    )
    total_label = rng.choice(["Total", "Invoice Total", "Grand Total", "Total Amount"])
    totals.append(f"{total_label}: {m(total)}")
    if paid:
        totals += [f"Amount Paid: {m(paid)}", f"Balance Due: {m(total - paid)}"]
    elif rng.random() < 0.3:
        totals.append(f"Previous Balance: {m(Decimal(rng.randint(100, 4000)))} (paid, thank you)")

    layout = rng.randrange(4)
    if layout == 0:  # classic table
        items = [f"{'Description':<38}{'Qty':>8}{'Unit':>14}{'Amount':>14}"]
        items += [
            f"{li['description']:<38}{_qty(li['quantity']):>8}{m(li['unit_price']):>14}{m(li['amount']):>14}"
            for li in lines
        ]
        body = vendor_block + ["", "INVOICE", *hdr, "", *bill_to, "", *items, "", *totals]
    elif layout == 1:  # key: value + bullets
        items = [
            f"- {li['description']}: {_qty(li['quantity'])} x {m(li['unit_price'])} = {m(li['amount'])}"
            for li in lines
        ]
        body = [f"From: {vendor}", *bill_to[:1], *hdr, "Items:", *items, *totals, vendor_block[2]]
    elif layout == 2:  # email narrative
        items = [
            f"  {i + 1}. {li['description']} -- qty {_qty(li['quantity'])} @ {m(li['unit_price'])}, "
            f"line total {m(li['amount'])}"
            for i, li in enumerate(lines)
        ]
        body = [
            f"Hello {customer} team,",
            f"Please find below invoice {inv_no} from {vendor} for recent work.",
            *[h for h in hdr if not h.endswith(inv_no)],
            "Charges:",
            *items,
            *totals,
            f"Questions? Call {_phone(rng)}. Remit to account {rng.randint(10**9, 10**10 - 1)}.",
        ]
    else:  # compact, OCR-ish pipes
        items = [
            f"| {li['description']} | {_qty(li['quantity'])} | {m(li['unit_price'])} | {m(li['amount'])} |"
            for li in lines
        ]
        body = [
            f"{vendor.upper()} | {vendor_block[1]}",
            " | ".join(hdr),
            bill_to[0],
            "| ITEM | QTY | PRICE | TOTAL |",
            *items,
            " / ".join(totals),
            f"Page 1 of 1 | Ref {rng.randint(100000, 999999)}",
        ]
    text = "\n".join(body)
    label = {
        "invoice_number": inv_no,
        "invoice_date": issued.isoformat(),
        "due_date": due.isoformat() if due else None,
        "vendor_name": vendor,
        "customer_name": customer,
        "po_number": po,
        "currency": currency,
        "subtotal": float(subtotal),
        "tax_amount": float(tax) if tax is not None else 0.0,  # exempt invoices print a 0.00 tax line
        "shipping_amount": float(shipping) if shipping else None,
        "total_amount": float(total),
        "line_items": [
            {
                "description": li["description"],
                "quantity": float(li["quantity"]),
                "unit_price": float(li["unit_price"]),
                "amount": float(li["amount"]),
            }
            for li in lines
        ],
    }
    return Invoice(text=text, label=label)


def _qty(q: Decimal) -> str:
    return f"{q:f}".rstrip("0").rstrip(".") if "." in f"{q:f}" else f"{q:f}"


def write_split(path: Path, n: int, seed: int, prefix: str) -> int:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for i in range(n):
            inv = make_invoice(rng, f"{prefix}{i:06d}")
            f.write(
                json.dumps(
                    {"id": f"{prefix}-{i:06d}", "text": inv.text, "label": inv.label}, ensure_ascii=False
                )
            )
            f.write("\n")
    tmp.replace(path)
    return n


def generate(out_dir: str | Path, *, n_train: int, n_eval: int, seed: int) -> dict[str, str]:
    out = Path(out_dir)
    train, ev = out / "train.jsonl", out / "eval.jsonl"
    # Independent streams and disjoint serial prefixes: no invoice text can appear in both splits.
    write_split(train, n_train, seed, "T")
    write_split(ev, n_eval, seed + 1_000_003, "E")
    return {"train": str(train), "eval": str(ev)}
