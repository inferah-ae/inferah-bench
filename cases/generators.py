"""
inferah-bench case generators: 28 deterministic cases, 7 failure types x 4
parameterizations, all on one order-grain data model:

    order_id, ts_day, period (p0/p1), country, city, order_type, category,
    gmv, user_id

Each case is a (DataFrame, label) pair. The label is the ground truth the
scorer compares against; the DataFrame is what gets loaded into Postgres
schema case_NN. Everything is seeded — same code, same bytes.

Domain is generic e-commerce; every name below is a fictional placeholder.
The generators extend the patterns of inferah_engine.synthetic (make_orders /
make_orders_unmapped / make_foodtech_deep): explicit per-segment row blocks
whose arithmetic you can check by eye, plus a seeded RNG only where a case
NEEDS noise (T6) — never to hide the injected cause.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
P0_DAYS = pd.date_range("2026-05-04", periods=7).date.tolist()   # Mon..Sun
P1_DAYS = pd.date_range("2026-05-11", periods=7).date.tolist()

COUNTRIES = ("Norvik", "Meridia", "Sundara")
CITIES = {  # 2 cities per country, disjoint names
    "Norvik": ("Aldgate", "Brimley"),
    "Meridia": ("Portvale", "Quarrytown"),
    "Sundara": ("Riversea", "Thornfield"),
}
CATEGORIES = ("electronics", "fashion", "grocery", "home")
PRICE = {"electronics": 80.0, "fashion": 40.0, "grocery": 15.0, "home": 55.0}

QUESTION = "Why did GMV change between period p0 and p1?"


class _Builder:
    """Accumulates order rows. One call = one homogeneous block of orders,
    spread deterministically across the period's days and a pool of users."""

    def __init__(self):
        self.rows = []
        self._oid = 0

    def add(self, period, country, city, category, order_type, n,
            price=None, days=None, user_pool=200, user_salt="", gmv_mult=1.0):
        days = days if days is not None else (P0_DAYS if period == "p0" else P1_DAYS)
        price = PRICE[category] if price is None else price
        for i in range(int(n)):
            self._oid += 1
            self.rows.append(dict(
                order_id=self._oid,
                ts_day=days[i % len(days)],
                period=period,
                country=country, city=city,
                order_type=order_type, category=category,
                gmv=round(price * gmv_mult, 2),
                user_id=f"{(country or 'xx')[:2].lower()}{user_salt}-{i % user_pool}",
            ))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _baseline(b: _Builder, period: str, scale: float = 1.0,
              promo_share: float = 0.2, skip=None, mult=None, user_pool=200):
    """The standard healthy week: every (country, city, category) cell gets
    100 orders (80 organic / 20 promo at promo price x0.8). ~7,200 orders,
    even axes, so any injected change stands out cleanly against it.

    skip:  set of (country, city, category) cells to leave out entirely
    mult:  {(country, city, category): volume multiplier} for per-cell shifts
    """
    skip = skip or set()
    mult = mult or {}
    for country in COUNTRIES:
        for city in CITIES[country]:
            for cat in CATEGORIES:
                if (country, city, cat) in skip:
                    continue
                m = scale * mult.get((country, city, cat), 1.0)
                n_org = round(100 * (1 - promo_share) * m)
                n_pro = round(100 * promo_share * m)
                b.add(period, country, city, cat, "organic", n_org,
                      user_pool=user_pool)
                b.add(period, country, city, cat, "promo", n_pro,
                      price=PRICE[cat] * 0.8, user_pool=user_pool)


def _label(case_id, ctype, action, drivers=None, abstain_reason=None,
           notes=""):
    return {
        "case_id": case_id,
        "type": ctype,
        "question": QUESTION,
        "expected": {
            "action": action,                      # explain | abstain | no_driver
            "drivers": drivers or [],
            "abstain_reason": abstain_reason,      # unmapped_dimension | data_gap | None
        },
        "notes": notes,
    }


def _driver(dimension=None, segment=None, factor=None, mechanism=None,
            share_of_move=1.0):
    # Scored fields only — dimension / segment / factor / mechanism / share.
    return {"dimension": dimension, "segment": segment,
            "factor": factor, "mechanism": mechanism,
            "share_of_move": share_of_move}


# ===================================================================== T1
# Segment drop (easy control): 100% of the move in one segment, volume.

def t1_segment(case_id, dimension, segment, direction):
    b = _Builder()
    _baseline(b, "p0")
    # cut (or boost) volume 60% in every cell belonging to the target segment
    factor = 0.4 if direction == "drop" else 1.6
    mult = {}
    for country in COUNTRIES:
        for city in CITIES[country]:
            for cat in CATEGORIES:
                hit = {"country": country, "city": city, "category": cat}[dimension] == segment
                if hit:
                    mult[(country, city, cat)] = factor
    _baseline(b, "p1", mult=mult)
    lbl = _label(case_id, "T1", "explain",
                 [_driver(dimension, segment, "orders", "volume", 1.0)],
                 notes=f"100% of the GMV {direction} is an order-volume move in "
                       f"{dimension}={segment}; every other segment is flat.")
    return b.frame(), lbl


# ===================================================================== T2
# Factor split: the move sits in ONE factor of GMV = buyers x freq x AOV.

def t2_orders_uniform(case_id):
    """Depth-2: orders fall ~15% uniformly everywhere, AOV flat."""
    b = _Builder()
    _baseline(b, "p0")
    _baseline(b, "p1", scale=0.85)
    lbl = _label(case_id, "T2", "explain",
                 [_driver(None, None, "orders", "volume", 1.0)],
                 notes="Order volume falls ~15% uniformly across every axis; "
                       "AOV is flat. The carrying factor is orders.")
    return b.frame(), lbl


def t2_aov_rate(case_id):
    """Depth-2: AOV falls via a real price cut, concentrated in organic orders
    (organic basket -15%, promo untouched); volume flat."""
    b = _Builder()
    _baseline(b, "p0")
    for country in COUNTRIES:
        for city in CITIES[country]:
            for cat in CATEGORIES:
                b.add("p1", country, city, cat, "organic", 80, gmv_mult=0.85)
                b.add("p1", country, city, cat, "promo", 20, price=PRICE[cat] * 0.8)
    lbl = _label(case_id, "T2", "explain",
                 [_driver(None, None, "aov", "rate", 1.0)],
                 notes="Orders flat; AOV falls because the organic basket got "
                       "~15% cheaper (a within-segment RATE move). Mix is stable.")
    return b.frame(), lbl


def t2_freq(case_id):
    """Depth-4: frequency falls — same buyers, each places ~50% fewer orders.
    The user pool is 50 per country, so even the reduced 50-order cells still
    touch every user: buyers flat, freq -50%, AOV flat."""
    b = _Builder()
    _baseline(b, "p0", promo_share=0.0, user_pool=50)
    _baseline(b, "p1", promo_share=0.0, scale=0.5, user_pool=50)
    lbl = _label(case_id, "T2", "explain",
                 [_driver(None, None, "freq", "volume", 1.0)],
                 notes="GMV falls 50% through orders; distinct buyers are "
                       "UNCHANGED (every user in the pool still orders), so "
                       "within orders = buyers x freq, frequency carries the "
                       "whole move. AOV is flat.")
    return b.frame(), lbl


# ===================================================================== T3
# Simpson / rate-vs-mix.

def t3_mix_order_type(case_id, direction="drop"):
    """Per-segment AOV is identical in both periods; the promo share moves
    (20% -> 50% on drop, 50% -> 20% on spike). Orders flat -> AOV is pure MIX."""
    b = _Builder()
    s0, s1 = (0.2, 0.5) if direction == "drop" else (0.5, 0.2)
    _baseline(b, "p0", promo_share=s0)
    _baseline(b, "p1", promo_share=s1)
    grew = "promo" if direction == "drop" else "organic"
    lbl = _label(case_id, "T3", "explain",
                 [_driver("order_type", grew, "aov", "mix", 1.0)],
                 notes=f"Within-segment AOV never changes (organic at list "
                       f"price, promo at 0.8x in BOTH periods); the "
                       f"{grew} share grew, so aggregate AOV moved by MIX only.")
    return b.frame(), lbl


def t3_mix_category(case_id):
    """Per-category AOV unchanged; order mix shifts from electronics (80) to
    grocery (15) with TOTAL orders flat -> aggregate AOV falls, pure mix."""
    b = _Builder()
    _baseline(b, "p0", promo_share=0.0)
    mult = {}
    for country in COUNTRIES:
        for city in CITIES[country]:
            mult[(country, city, "electronics")] = 0.5   # -50 orders per cell
            mult[(country, city, "grocery")] = 1.5       # +50 orders per cell
    _baseline(b, "p1", promo_share=0.0, mult=mult)
    lbl = _label(case_id, "T3", "explain",
                 [_driver("category", "grocery", "aov", "mix", 1.0)],
                 notes="Every category's own AOV is identical across periods "
                       "and total orders are flat; the order mix shifted from "
                       "electronics to grocery (cheap), so AOV fell by MIX.")
    return b.frame(), lbl


def t3_reverse_simpson(case_id):
    """Aggregate exactly flat, but inside: the organic basket fell 7.5% and
    the promo share fell 50%->20% in perfect compensation. The deceptive
    'nothing happened' aggregate.

    Arithmetic per cell (price p): p0 = 50p + 50*0.8p = 90p;
    p1 = 80*0.925p + 20*0.8p = 74p + 16p = 90p."""
    b = _Builder()
    _baseline(b, "p0", promo_share=0.5)
    for country in COUNTRIES:
        for city in CITIES[country]:
            for cat in CATEGORIES:
                b.add("p1", country, city, cat, "organic", 80, gmv_mult=0.925)
                b.add("p1", country, city, cat, "promo", 20, price=PRICE[cat] * 0.8)
    lbl = _label(case_id, "T3", "no_driver",
                 notes="Reverse Simpson: aggregate GMV is flat (~0%) yet the "
                       "organic basket fell 7.5% and the promo share fell "
                       "50%->20% in exact compensation. Correct answer: no "
                       "significant aggregate driver — the stability of the "
                       "aggregate is deceptive (compensating effects).")
    return b.frame(), lbl


# ===================================================================== T4
# Off-tree / unmapped: the true driver is a dimension the data doesn't carry.

def t4_unmapped_geo(case_id, frac_unmapped=1.0):
    """An unmapped acquisition source: a block of orders with country AND city
    NULL (no geo enrichment), spread evenly across categories, collapses.
    frac_unmapped=1.0 puts 100% of the move in the NULL rows; 0.6 leaves 40%
    as a genuine volume drop in country=Sundara. Either way a geo split can't
    reconcile, and the category split sees only an even, non-dominant drop."""
    b = _Builder()
    _baseline(b, "p0")

    def add_null(period, n_per_cat):
        for cat in CATEGORIES:
            b.add(period, None, None, cat, "organic", n_per_cat,
                  price=50.0, user_salt="x")

    add_null("p0", 500)                                   # 2000 x 50 = 100,000
    mult = {}
    if frac_unmapped < 1.0:
        for city in CITIES["Sundara"]:
            for cat in CATEGORIES:
                mult[("Sundara", city, cat)] = 0.45       # mapped: -20,064
    _baseline(b, "p1", mult=mult)
    # frac 1.0: NULL -90,000 = 100% of the move; frac 0.6: NULL -30,000 vs
    # mapped -20,064 -> NULL holds ~60% of the total move.
    add_null("p1", 50 if frac_unmapped == 1.0 else 350)
    lbl = _label(case_id, "T4", "abstain", abstain_reason="unmapped_dimension",
                 notes=f"~{int(frac_unmapped*100)}% of the move sits in rows "
                       f"with country/city IS NULL (an unmapped source) — no "
                       f"modeled dimension accounts for it; a geo split can't "
                       f"reconcile. The honest answer is abstain.")
    return b.frame(), lbl


def t4_unmapped_category(case_id):
    """The collapsing block has category=NULL but real geo, spread evenly
    across all six cities — so geo splits show a uniform non-dominant drop and
    the category split leaves the move unexplained."""
    b = _Builder()
    _baseline(b, "p0")

    def add_null(period, n_per_city):
        for country in COUNTRIES:
            for city in CITIES[country]:
                b.add(period, country, city, None, "organic", n_per_city,
                      price=50.0, user_salt="x")

    add_null("p0", 330)                                   # 1980 x 50 = 99,000
    _baseline(b, "p1")
    add_null("p1", 33)                                    # -89,100, 100% of move
    lbl = _label(case_id, "T4", "abstain", abstain_reason="unmapped_dimension",
                 notes="100% of the move sits in rows with category IS NULL, "
                       "spread evenly across every city — the category split "
                       "can't reconcile and no geo segment dominates. The "
                       "honest answer is abstain.")
    return b.frame(), lbl


def t4_external_uniform(case_id):
    """The cause is entirely OUTSIDE the data: a market-wide price-pressure
    event cuts every order's basket ~10% in every country/city/category and
    in BOTH order types. Volume flat, mix flat, no NULLs. The promo share is
    0.55 so the per-type rate contributions are BALANCED (organic 45 x 1.0p
    vs promo 55 x 0.8p) — no order type concentrates the rate effect."""
    b = _Builder()
    _baseline(b, "p0", promo_share=0.55)
    for country in COUNTRIES:
        for city in CITIES[country]:
            for cat in CATEGORIES:
                b.add("p1", country, city, cat, "organic", 45, gmv_mult=0.9)
                b.add("p1", country, city, cat, "promo", 55,
                      price=PRICE[cat] * 0.8, gmv_mult=0.9)
    lbl = _label(case_id, "T4", "abstain", abstain_reason="unmapped_dimension",
                 notes="AOV falls a uniform ~10% inside EVERY segment of every "
                       "axis with volume and mix flat — the move is equally "
                       "present everywhere, so no column localizes it. The "
                       "cause is an external market event; honest answer: "
                       "abstain (diffuse, not residual).")
    return b.frame(), lbl


# ===================================================================== T5
# Compound: two independent drivers at once.

def t5_compound(case_id, share_a, mech_b="volume"):
    """Driver A: order-volume drop in country=Norvik sized to share_a of the
    total move. Driver B (1-share_a): volume drop in category=fashion
    OUTSIDE Norvik (mech_b='volume'), or an order_type mix shift inside
    country=Sundara only (mech_b='mix')."""
    b = _Builder()
    promo = 0.2 if mech_b == "volume" else 0.5
    _baseline(b, "p0", promo_share=promo)

    # Multipliers solved analytically from the standard grid so the two
    # drivers land on share_a/(1-share_a) of the total move; the sanity
    # script (and a unit test) verify the realized shares.
    cut_a = {("volume", 0.7): 0.50, ("volume", 0.5): 0.70,
             ("mix", 0.7): 0.8185, ("mix", 0.5): 0.9222}[(mech_b, share_a)]
    mult = {}
    for city in CITIES["Norvik"]:
        for cat in CATEGORIES:
            mult[("Norvik", city, cat)] = cut_a

    if mech_b == "volume":
        cut_b = {0.7: 0.49, 0.5: 0.2875}[share_a]
        for country in ("Meridia", "Sundara"):
            for city in CITIES[country]:
                mult[(country, city, "fashion")] = cut_b
        _baseline(b, "p1", promo_share=promo, mult=mult)
        driver_b = _driver("category", "fashion", "orders", "volume",
                           round(1 - share_a, 2))
    else:
        # mix shift inside Sundara only: promo share 0.5 -> 0.85 there
        skip = {("Sundara", c, k) for c in CITIES["Sundara"] for k in CATEGORIES}
        _baseline(b, "p1", promo_share=promo, mult=mult, skip=skip)
        for city in CITIES["Sundara"]:
            for cat in CATEGORIES:
                b.add("p1", "Sundara", city, cat, "organic", 15)
                b.add("p1", "Sundara", city, cat, "promo", 85,
                      price=PRICE[cat] * 0.8)
        driver_b = _driver("country", "Sundara", "aov", "mix",
                           round(1 - share_a, 2))

    lbl = _label(case_id, "T5", "explain",
                 [_driver("country", "Norvik", "orders", "volume", share_a),
                  driver_b],
                 notes=f"TWO independent drivers: ~{int(share_a*100)}% of the "
                       f"move is an order-volume drop in Norvik; the rest is "
                       f"{'a fashion volume drop elsewhere' if mech_b=='volume' else 'an order_type mix shift inside Sundara'}. "
                       f"A complete answer names both with shares.")
    return b.frame(), lbl


# ===================================================================== T6
# Noise / borderline: the move is within historical noise.

def t6_noise(case_id, seed, tiny_effect=None, target_delta=-0.011):
    """Daily volumes wobble (sigma ~6% per day-cell); the p1 aggregate lands
    within ~1 sigma of p0 (|delta| ~1.1% < the 2% significance bar). With
    tiny_effect=('country', 'Meridia', 0.97) a small REAL effect hides inside
    the noise — the overconfidence trap: it's there, but not provable."""
    rng = np.random.default_rng(seed)
    b = _Builder()

    def noisy_week(period, base_mult):
        for country in COUNTRIES:
            for city in CITIES[country]:
                for cat in CATEGORIES:
                    m = base_mult
                    if tiny_effect and period == "p1":
                        dim, seg, f = tiny_effect
                        row = {"country": country, "city": city, "category": cat}
                        if row[dim] == seg:
                            m = m * f
                    days = P0_DAYS if period == "p0" else P1_DAYS
                    for d in days:
                        n = max(1, round((100 / 7) * m * rng.normal(1.0, 0.06)))
                        b.add(period, country, city, cat, "organic",
                              round(n * 0.8), days=[d])
                        b.add(period, country, city, cat, "promo",
                              max(0, n - round(n * 0.8)),
                              price=PRICE[cat] * 0.8, days=[d])

    noisy_week("p0", 1.0)
    noisy_week("p1", 1.0 + target_delta)
    lbl = _label(case_id, "T6", "no_driver",
                 notes="The week-over-week move is within historical daily "
                       "noise (|delta| < 2 sigma of the daily series, < the "
                       "2% significance bar)."
                       + (" A tiny real effect (~3% in one country) hides in "
                          "the noise — still not significant; claiming a "
                          "driver is overconfidence." if tiny_effect else
                          " Pure noise: no driver exists at all."))
    return b.frame(), lbl


# ===================================================================== T7
# Data gap / reconcile fail: p1 is missing rows; decomposition is built on
# incomplete data and the correct answer is abstain(data_gap).

def _daily_grid(b, period, days, org_per_day=12, promo_per_day=3,
                only_cat=None):
    """Day-by-day baseline: every cell gets org_per_day+promo_per_day orders
    PER DAY, so removing a day genuinely removes that day's volume."""
    for country in COUNTRIES:
        for city in CITIES[country]:
            for cat in CATEGORIES:
                if only_cat and cat != only_cat:
                    continue
                for d in days:
                    b.add(period, country, city, cat, "organic",
                          org_per_day, days=[d])
                    b.add(period, country, city, cat, "promo", promo_per_day,
                          price=PRICE[cat] * 0.8, days=[d])


def t7_missing_day(case_id, missing_idx):
    """One whole day vanishes from p1 (a failed pipeline load). GMV 'falls'
    ~14% but it's an artifact: p1 has 6 days of data vs 7, evenly across
    every segment."""
    b = _Builder()
    _daily_grid(b, "p0", P0_DAYS)
    _daily_grid(b, "p1", [d for i, d in enumerate(P1_DAYS) if i != missing_idx])
    lbl = _label(case_id, "T7", "abstain", abstain_reason="data_gap",
                 notes=f"p1 contains only 6 distinct days (day index "
                       f"{missing_idx} of the week is absent everywhere) — "
                       f"the 'drop' is a missing data load, not demand. "
                       f"Honest answer: abstain, data incomplete.")
    return b.frame(), lbl


def t7_missing_segment(case_id, dimension, segment):
    """One source vanishes: every p1 row of a whole segment is absent (zero
    rows at all — not low volume). Looks like a 100% segment drop; the
    complete absence is the data-gap signature."""
    b = _Builder()
    _baseline(b, "p0")
    skip = set()
    for country in COUNTRIES:
        for city in CITIES[country]:
            for cat in CATEGORIES:
                if {"country": country, "city": city,
                        "category": cat}[dimension] == segment:
                    skip.add((country, city, cat))
    _baseline(b, "p1", skip=skip)
    lbl = _label(case_id, "T7", "abstain", abstain_reason="data_gap",
                 notes=f"{dimension}={segment} has ZERO rows in p1 (not a "
                       f"decline — a complete absence). A real demand drop "
                       f"leaves a remnant; total absence means the source "
                       f"didn't load. Honest answer: abstain, data gap.")
    return b.frame(), lbl


def t7_partial_segment(case_id):
    """category=grocery stops loading midway through p1: present for the first
    3 days, absent for the last 4 — a within-period truncation."""
    b = _Builder()
    _daily_grid(b, "p0", P0_DAYS)
    for cat in CATEGORIES:
        days = P1_DAYS[:3] if cat == "grocery" else P1_DAYS
        _daily_grid(b, "p1", days, only_cat=cat)
    lbl = _label(case_id, "T7", "abstain", abstain_reason="data_gap",
                 notes="grocery rows exist only for the first 3 days of p1 "
                       "and stop dead mid-period — a truncated load, not a "
                       "demand cliff. Honest answer: abstain, data gap.")
    return b.frame(), lbl


def t2_buyers(case_id):
    """Depth-4: 25% of buyers churn entirely; survivors keep frequency and
    basket -> orders -25%, buyers -25%, freq flat, AOV flat. p0 cells of 100
    orders cover users 0..99; p1 cells of 75 orders cover users 0..74."""
    b = _Builder()
    _baseline(b, "p0", promo_share=0.0, user_pool=100)
    _baseline(b, "p1", promo_share=0.0, scale=0.75, user_pool=100)
    lbl = _label(case_id, "T2", "explain",
                 [_driver(None, None, "buyers", "volume", 1.0)],
                 notes="A quarter of buyers churn outright (users 75..99 of "
                       "every pool vanish); the survivors' frequency and "
                       "basket are unchanged, so within GMV = buyers x freq "
                       "x AOV the buyers factor carries ~100% of the drop.")
    return b.frame(), lbl


# ================================================================ registry
def all_cases():
    """[(case_id, builder_fn)] in fixed order — case_01..case_28."""
    specs = [
        # T1 — segment drop (control)
        lambda cid: t1_segment(cid, "country", "Meridia", "drop"),
        lambda cid: t1_segment(cid, "city", "Portvale", "drop"),
        lambda cid: t1_segment(cid, "category", "electronics", "drop"),
        lambda cid: t1_segment(cid, "country", "Norvik", "spike"),
        # T2 — factor split
        t2_orders_uniform,
        t2_aov_rate,
        t2_buyers,
        t2_freq,
        # T3 — Simpson / rate-vs-mix
        lambda cid: t3_mix_order_type(cid, "drop"),
        t3_mix_category,
        t3_reverse_simpson,
        lambda cid: t3_mix_order_type(cid, "spike"),
        # T4 — off-tree / unmapped
        lambda cid: t4_unmapped_geo(cid, 1.0),
        lambda cid: t4_unmapped_geo(cid, 0.6),
        t4_external_uniform,
        t4_unmapped_category,
        # T5 — compound
        lambda cid: t5_compound(cid, 0.7, "volume"),
        lambda cid: t5_compound(cid, 0.5, "volume"),
        lambda cid: t5_compound(cid, 0.7, "mix"),
        lambda cid: t5_compound(cid, 0.5, "mix"),
        # T6 — noise / borderline
        lambda cid: t6_noise(cid, seed=601),
        lambda cid: t6_noise(cid, seed=602, tiny_effect=("country", "Meridia", 0.97),
                             target_delta=-0.003),
        lambda cid: t6_noise(cid, seed=603, target_delta=0.009),
        lambda cid: t6_noise(cid, seed=604, tiny_effect=("category", "home", 0.96),
                             target_delta=-0.004),
        # T7 — data gap
        lambda cid: t7_missing_day(cid, 6),
        lambda cid: t7_missing_segment(cid, "country", "Sundara"),
        lambda cid: t7_missing_day(cid, 2),
        t7_partial_segment,
    ]
    return [(f"case_{i+1:02d}", fn) for i, fn in enumerate(specs)]


def build_case(case_id):
    fn = dict(all_cases())[case_id]
    df, lbl = fn(case_id)
    df = df.copy()
    lbl["headline_numbers"] = _headline(df)
    return df, lbl


def _headline(df) -> dict:
    g0 = df[df.period == "p0"].gmv.sum()
    g1 = df[df.period == "p1"].gmv.sum()
    return {"metric_delta_pct": round((g1 - g0) / g0 * 100, 1)}


def build_all():
    """[(case_id, df, label)] for all 28 cases."""
    out = []
    for case_id, _ in all_cases():
        df, lbl = build_case(case_id)
        out.append((case_id, df, lbl))
    return out


# ================================================================ seeding
DDL = """
DROP SCHEMA IF EXISTS {schema} CASCADE;
CREATE SCHEMA {schema};
CREATE TABLE {schema}.orders (
    order_id   integer       NOT NULL,
    ts_day     date          NOT NULL,
    period     text          NOT NULL,
    country    text,
    city       text,
    order_type text          NOT NULL,
    category   text,
    gmv        numeric(12,2) NOT NULL,
    user_id    text          NOT NULL
);
"""


def seed_postgres(engine, cases=None, verbose=True):
    """Idempotently (re)create case_NN schemas. `cases` limits to a subset of
    case ids; default seeds all 28."""
    from sqlalchemy import text
    if cases is None:
        built = build_all()
    else:
        built = [(cid, *build_case(cid)) for cid in cases]
    labels = []
    for case_id, df, lbl in built:
        with engine.begin() as conn:
            for stmt in DDL.format(schema=case_id).split(";"):
                if stmt.strip():
                    conn.execute(text(stmt))
        df.to_sql("orders", engine, schema=case_id, if_exists="append",
                  index=False, method="multi", chunksize=5000)
        labels.append(lbl)
        if verbose:
            print(f"  {case_id} [{lbl['type']}]  {len(df):>6,} rows  "
                  f"delta {lbl['headline_numbers']['metric_delta_pct']:+.1f}%")
    return labels
