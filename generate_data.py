"""
generate_data.py
================
Generates a realistic 3-year e-commerce SQLite database for the
E-commerce Sales Intelligence Dashboard project.

Tables created:
  - customers   (10,000 rows)
  - products    (500 rows)
  - orders      (50,000 rows)
  - order_items (75,000 rows)
  - events      (200,000 rows)  ← funnel events (view/cart/checkout/purchase)

Usage:
  python generate_data.py              # full dataset
  python generate_data.py --rows 1000  # smaller dataset for CI/testing

Dependencies:
  pip install faker pandas numpy sqlalchemy
"""

import argparse
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
fake.seed_instance(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH       = Path("data/ecommerce.db")
START_DATE    = datetime(2022, 1, 1)
END_DATE      = datetime(2024, 12, 31)
N_CUSTOMERS   = 10_000
N_PRODUCTS    = 500
N_ORDERS      = 50_000

# ── Indian city/state pairs with tier classification ───────────────────────────
CITIES = [
    ("Mumbai",       "Maharashtra",   "Tier-1"),
    ("Delhi",        "Delhi",         "Tier-1"),
    ("Bengaluru",    "Karnataka",     "Tier-1"),
    ("Hyderabad",    "Telangana",     "Tier-1"),
    ("Chennai",      "Tamil Nadu",    "Tier-1"),
    ("Kolkata",      "West Bengal",   "Tier-1"),
    ("Pune",         "Maharashtra",   "Tier-2"),
    ("Ahmedabad",    "Gujarat",       "Tier-2"),
    ("Jaipur",       "Rajasthan",     "Tier-2"),
    ("Surat",        "Gujarat",       "Tier-2"),
    ("Lucknow",      "Uttar Pradesh", "Tier-2"),
    ("Chandigarh",   "Punjab",        "Tier-2"),
    ("Bhopal",       "Madhya Pradesh","Tier-2"),
    ("Patna",        "Bihar",         "Tier-3"),
    ("Indore",       "Madhya Pradesh","Tier-2"),
    ("Nagpur",       "Maharashtra",   "Tier-2"),
    ("Coimbatore",   "Tamil Nadu",    "Tier-2"),
    ("Kochi",        "Kerala",        "Tier-2"),
    ("Agra",         "Uttar Pradesh", "Tier-3"),
    ("Varanasi",     "Uttar Pradesh", "Tier-3"),
    ("Meerut",       "Uttar Pradesh", "Tier-3"),
    ("Nashik",       "Maharashtra",   "Tier-3"),
    ("Vijayawada",   "Andhra Pradesh","Tier-3"),
    ("Jodhpur",      "Rajasthan",     "Tier-3"),
    ("Madurai",      "Tamil Nadu",    "Tier-3"),
]
CITY_WEIGHTS = [
    0.09, 0.09, 0.09, 0.07, 0.07, 0.06,
    0.04, 0.04, 0.04, 0.03, 0.03, 0.03,
    0.03, 0.02, 0.03, 0.03, 0.03, 0.03,
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
]

# ── Product catalogue ──────────────────────────────────────────────────────────
CATEGORIES = {
    "Electronics":   {"weight": 0.22, "price_range": (2000,  80000), "margin": 0.12},
    "Fashion":       {"weight": 0.20, "price_range": (300,    8000), "margin": 0.45},
    "Home & Kitchen":{"weight": 0.15, "price_range": (500,   15000), "margin": 0.30},
    "Books":         {"weight": 0.10, "price_range": (150,    2000), "margin": 0.35},
    "Beauty":        {"weight": 0.10, "price_range": (200,    5000), "margin": 0.55},
    "Sports":        {"weight": 0.08, "price_range": (500,   12000), "margin": 0.28},
    "Toys":          {"weight": 0.07, "price_range": (300,    6000), "margin": 0.40},
    "Grocery":       {"weight": 0.08, "price_range": (50,     2000), "margin": 0.18},
}

PRODUCT_ADJECTIVES = [
    "Premium", "Classic", "Deluxe", "Pro", "Ultra", "Smart", "Eco",
    "Compact", "Wireless", "Portable", "Slim", "Heavy-Duty", "Organic",
]
PRODUCT_NOUNS = {
    "Electronics":    ["Earbuds", "Smartwatch", "Laptop Stand", "USB Hub", "Webcam",
                       "Power Bank", "Keyboard", "Mouse", "Speaker", "Charger"],
    "Fashion":        ["Kurta", "Jeans", "Saree", "T-Shirt", "Sneakers",
                       "Handbag", "Watch", "Sunglasses", "Jacket", "Salwar Suit"],
    "Home & Kitchen": ["Mixer", "Pressure Cooker", "Air Fryer", "Bedsheet",
                       "Pillow", "Curtains", "Lamp", "Organizer", "Tiffin Box", "Mop"],
    "Books":          ["Novel", "Self-Help Book", "Textbook", "Comic", "Cookbook",
                       "Biography", "Children's Book", "Stationery Set", "Planner", "Atlas"],
    "Beauty":         ["Face Cream", "Lipstick", "Shampoo", "Serum", "Sunscreen",
                       "Foundation", "Kajal", "Body Lotion", "Hair Oil", "Perfume"],
    "Sports":         ["Yoga Mat", "Dumbbell Set", "Cricket Bat", "Badminton Racket",
                       "Running Shoes", "Gym Bag", "Cycle Helmet", "Resistance Band", "Skipping Rope", "Football"],
    "Toys":           ["Board Game", "Lego Set", "Doll", "Remote Car", "Puzzle",
                       "Art Kit", "Building Blocks", "Soft Toy", "Science Kit", "Action Figure"],
    "Grocery":        ["Olive Oil", "Protein Bar", "Muesli", "Herbal Tea",
                       "Ghee", "Dry Fruits Mix", "Quinoa", "Dark Chocolate", "Instant Oats", "Honey"],
}

# ── Acquisition channels ───────────────────────────────────────────────────────
CHANNELS = ["Organic Search", "Social Media", "Paid Ads", "Referral",
            "Email Campaign", "App", "Direct"]
CHANNEL_WEIGHTS = [0.25, 0.22, 0.20, 0.12, 0.10, 0.07, 0.04]

# ── Seasonality multipliers by month ──────────────────────────────────────────
# Oct/Nov = Diwali boom, Jan = post-festival dip, Aug = Independence Day
MONTHLY_MULTIPLIERS = {
    1: 0.70, 2: 0.75, 3: 0.85, 4: 0.88,
    5: 0.90, 6: 0.92, 7: 0.88, 8: 1.05,
    9: 0.95, 10: 1.30, 11: 1.55, 12: 1.10,
}


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def weighted_random(items, weights):
    return random.choices(items, weights=weights, k=1)[0]


# ── 1. Customers ───────────────────────────────────────────────────────────────
def generate_customers(n: int) -> pd.DataFrame:
    print(f"  Generating {n:,} customers...")
    rows = []
    # Spread signups across 3 years with slight growth each year
    signup_weights = []
    for i in range(n):
        days_in_range = (END_DATE - START_DATE).days
        # More customers signed up in later periods (growth curve)
        day_offset = int(np.random.beta(1.5, 1.0) * days_in_range)
        signup_date = START_DATE + timedelta(days=day_offset)

        city_info    = weighted_random(CITIES, CITY_WEIGHTS)
        city, state, tier = city_info
        channel      = weighted_random(CHANNELS, CHANNEL_WEIGHTS)
        age          = int(np.random.normal(32, 9))
        age          = max(18, min(65, age))
        gender       = random.choices(["Male", "Female", "Other"],
                                      weights=[0.52, 0.45, 0.03])[0]

        # Churn propensity — Tier-3 cities and older customers churn slightly more
        base_churn = 0.25
        if tier == "Tier-3": base_churn += 0.10
        if tier == "Tier-1": base_churn -= 0.05
        if channel == "Referral": base_churn -= 0.08
        if channel == "Paid Ads": base_churn += 0.05

        rows.append({
            "customer_id":     i + 1,
            "name":            fake.name(),
            "email":           fake.email(),
            "phone":           fake.phone_number(),
            "city":            city,
            "state":           state,
            "city_tier":       tier,
            "signup_date":     signup_date.date().isoformat(),
            "channel":         channel,
            "age":             age,
            "gender":          gender,
            "churn_propensity": round(min(0.9, max(0.05, base_churn + random.gauss(0, 0.08))), 3),
        })
    return pd.DataFrame(rows)


# ── 2. Products ────────────────────────────────────────────────────────────────
def generate_products(n: int) -> pd.DataFrame:
    print(f"  Generating {n:,} products...")
    cat_names = list(CATEGORIES.keys())
    cat_weights = [CATEGORIES[c]["weight"] for c in cat_names]
    rows = []
    used_names = set()

    for i in range(n):
        cat_name   = weighted_random(cat_names, cat_weights)
        cat        = CATEGORIES[cat_name]
        pmin, pmax = cat["price_range"]
        # Lognormal price distribution within range
        price_raw  = np.random.lognormal(mean=np.log((pmin+pmax)/2), sigma=0.4)
        price      = round(max(pmin, min(pmax, price_raw)), -1)  # round to nearest 10

        adj  = random.choice(PRODUCT_ADJECTIVES)
        noun = random.choice(PRODUCT_NOUNS[cat_name])
        name = f"{adj} {noun}"
        # De-duplicate by appending a suffix
        suffix, attempt = "", 0
        while f"{name}{suffix}" in used_names:
            attempt += 1
            suffix  = f" {attempt}"
        name += suffix
        used_names.add(name)

        # Higher-priced products in a category are "premium" — lower return rate
        return_rate = round(random.uniform(0.02, 0.12), 3)
        rating      = round(min(5.0, max(3.0, random.gauss(4.1, 0.5))), 1)
        stock       = random.randint(0, 500)

        rows.append({
            "product_id":   i + 1,
            "name":         name,
            "category":     cat_name,
            "price":        float(price),
            "cost":         round(float(price) * (1 - cat["margin"]), 2),
            "margin_pct":   round(cat["margin"] * 100, 1),
            "rating":       rating,
            "return_rate":  return_rate,
            "stock":        stock,
            "launched_on":  (START_DATE - timedelta(days=random.randint(0, 365))).date().isoformat(),
        })
    return pd.DataFrame(rows)


# ── 3. Orders + Order Items ────────────────────────────────────────────────────
def generate_orders(n: int,
                    customers: pd.DataFrame,
                    products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"  Generating {n:,} orders + line items...")

    cust_ids    = customers["customer_id"].tolist()
    cust_signup = dict(zip(customers["customer_id"], customers["signup_date"]))
    cust_churn  = dict(zip(customers["customer_id"], customers["churn_propensity"]))
    prod_ids    = products["product_id"].tolist()
    prod_price  = dict(zip(products["product_id"], products["price"]))
    prod_cat    = dict(zip(products["product_id"], products["category"]))

    # Customers with high churn propensity stop ordering after ~6 months
    PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking",
                       "Cash on Delivery", "EMI", "Wallet"]
    PAY_WEIGHTS     = [0.38, 0.20, 0.18, 0.08, 0.08, 0.05, 0.03]

    STATUS_CHOICES  = ["delivered", "delivered", "delivered", "delivered",
                       "returned", "cancelled"]
    STATUS_WEIGHTS  = [0.78, 0.78, 0.78, 0.78, 0.10, 0.12]  # normalized below
    # Simplify: delivered 78%, returned 10%, cancelled 12%
    STATUSES        = ["delivered", "returned", "cancelled"]
    STAT_W          = [0.78, 0.10, 0.12]

    orders, items = [], []
    order_id = 1

    # Spread orders across date range with seasonality
    total_days = (END_DATE - START_DATE).days

    # Pre-generate a distribution of order dates weighted by seasonality
    all_days = [START_DATE + timedelta(days=d) for d in range(total_days + 1)]
    day_weights = []
    for d in all_days:
        m_mult  = MONTHLY_MULTIPLIERS[d.month]
        # Weekend boost
        w_mult  = 1.15 if d.weekday() >= 5 else 1.0
        day_weights.append(m_mult * w_mult)

    order_dates = random.choices(all_days, weights=day_weights, k=n)
    order_dates.sort()

    # Assign customers — repeat buyers cluster
    # ~20% of customers make 80% of orders (power law)
    power_weights = np.random.pareto(1.5, size=len(cust_ids)) + 1
    power_weights /= power_weights.sum()
    sampled_custs = np.random.choice(cust_ids, size=n, p=power_weights)

    item_id = 1
    for i in range(n):
        cid        = int(sampled_custs[i])
        order_date = order_dates[i]

        # Customer must have signed up before this order
        signup = datetime.fromisoformat(cust_signup[cid])
        if order_date < signup:
            order_date = signup + timedelta(days=random.randint(1, 30))
        if order_date > END_DATE:
            order_date = END_DATE

        # High-churn customers: 40% chance they stop ordering after 6 months
        months_active = (order_date - signup).days / 30
        if months_active > 6 and random.random() < cust_churn[cid] * 0.4:
            # Skip this order (simulate churn)
            continue

        n_items      = random.choices([1, 2, 3, 4, 5],
                                      weights=[0.50, 0.25, 0.13, 0.08, 0.04])[0]
        sampled_prods = random.sample(prod_ids, min(n_items, len(prod_ids)))
        payment      = weighted_random(PAYMENT_METHODS, PAY_WEIGHTS)
        status       = weighted_random(STATUSES, STAT_W)

        # Discount: 20% of orders get a coupon (Diwali season: 35%)
        disc_chance  = 0.35 if order_date.month in (10, 11) else 0.20
        discount_pct = random.choice([0, 0, 5, 10, 15, 20]) if random.random() < disc_chance else 0

        order_value  = 0.0
        for pid in sampled_prods:
            qty   = random.choices([1, 2, 3], weights=[0.70, 0.22, 0.08])[0]
            price = prod_price[pid]
            disc  = round(price * discount_pct / 100, 2)
            line  = round((price - disc) * qty, 2)
            order_value += line
            items.append({
                "item_id":     item_id,
                "order_id":    order_id,
                "product_id":  pid,
                "quantity":    qty,
                "unit_price":  price,
                "discount":    disc,
                "line_total":  line,
            })
            item_id += 1

        delivery_days = random.randint(2, 9) if status != "cancelled" else 0
        orders.append({
            "order_id":      order_id,
            "customer_id":   cid,
            "order_date":    order_date.date().isoformat(),
            "delivery_date": (order_date + timedelta(days=delivery_days)).date().isoformat()
                             if delivery_days > 0 else None,
            "status":        status,
            "payment_method":payment,
            "discount_pct":  discount_pct,
            "order_value":   round(order_value, 2),
            "is_first_order":0,  # filled below
        })
        order_id += 1

    orders_df = pd.DataFrame(orders)
    items_df  = pd.DataFrame(items)

    # Mark first orders
    first_orders = orders_df.groupby("customer_id")["order_id"].min()
    orders_df["is_first_order"] = orders_df["order_id"].isin(first_orders).astype(int)

    print(f"    → {len(orders_df):,} orders generated (after churn simulation)")
    print(f"    → {len(items_df):,} order line items")
    return orders_df, items_df


# ── 4. Funnel Events ───────────────────────────────────────────────────────────
def generate_events(orders: pd.DataFrame,
                    customers: pd.DataFrame,
                    n_sessions: int = 200_000) -> pd.DataFrame:
    """
    Simulates user funnel events: view → add_to_cart → checkout → purchase.
    Only a fraction of sessions result in a purchase (which ties to an order).
    """
    print(f"  Generating {n_sessions:,} funnel events...")

    STAGES = ["view", "add_to_cart", "checkout", "purchase"]
    # Conversion rates at each step: view→cart 35%, cart→checkout 55%, checkout→purchase 72%
    CONV   = [1.0, 0.35, 0.192, 0.138]  # cumulative: these get ~13.8% overall conversion

    cust_ids   = customers["customer_id"].tolist()
    order_ids  = orders["order_id"].tolist()
    order_dates= dict(zip(orders["order_id"], orders["order_date"]))

    events, used_orders = [], set()
    session_id = 1

    total_days = (datetime(2024, 12, 31) - START_DATE).days
    all_days   = [START_DATE + timedelta(days=d) for d in range(total_days + 1)]
    day_weights = [MONTHLY_MULTIPLIERS[d.month] * (1.12 if d.weekday() >= 5 else 1.0)
                   for d in all_days]

    for _ in range(n_sessions):
        sess_date = random.choices(all_days, weights=day_weights, k=1)[0]
        cust_id   = random.choice(cust_ids)
        session_ts = sess_date + timedelta(
            hours=random.randint(8, 23),
            minutes=random.randint(0, 59),
        )
        max_stage = 0
        for s, stage in enumerate(STAGES):
            if random.random() > CONV[s]:
                break
            max_stage = s

            ts = session_ts + timedelta(minutes=random.randint(0, 8) * s)
            ev = {
                "event_id":   len(events) + 1,
                "session_id": session_id,
                "customer_id":cust_id,
                "event_type": stage,
                "event_time": ts.isoformat(sep=" "),
                "order_id":   None,
            }
            # Tie purchase event to a real order if possible
            if stage == "purchase" and order_ids:
                oid = random.choice(order_ids)
                if oid not in used_orders:
                    ev["order_id"] = oid
                    used_orders.add(oid)
            events.append(ev)

        session_id += 1

    return pd.DataFrame(events)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_to_sqlite(customers, products, orders, items, events, db_path: Path):
    print(f"\n  Writing to {db_path} ...")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # ── Schema ──────────────────────────────────────────────────────────────
    cur.executescript("""
    CREATE TABLE customers (
        customer_id      INTEGER PRIMARY KEY,
        name             TEXT    NOT NULL,
        email            TEXT    NOT NULL,
        phone            TEXT,
        city             TEXT,
        state            TEXT,
        city_tier        TEXT,
        signup_date      TEXT,
        channel          TEXT,
        age              INTEGER,
        gender           TEXT,
        churn_propensity REAL
    );

    CREATE TABLE products (
        product_id   INTEGER PRIMARY KEY,
        name         TEXT    NOT NULL,
        category     TEXT    NOT NULL,
        price        REAL    NOT NULL,
        cost         REAL,
        margin_pct   REAL,
        rating       REAL,
        return_rate  REAL,
        stock        INTEGER,
        launched_on  TEXT
    );

    CREATE TABLE orders (
        order_id        INTEGER PRIMARY KEY,
        customer_id     INTEGER NOT NULL,
        order_date      TEXT    NOT NULL,
        delivery_date   TEXT,
        status          TEXT,
        payment_method  TEXT,
        discount_pct    INTEGER DEFAULT 0,
        order_value     REAL    NOT NULL,
        is_first_order  INTEGER DEFAULT 0,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );

    CREATE TABLE order_items (
        item_id     INTEGER PRIMARY KEY,
        order_id    INTEGER NOT NULL,
        product_id  INTEGER NOT NULL,
        quantity    INTEGER NOT NULL,
        unit_price  REAL    NOT NULL,
        discount    REAL    DEFAULT 0,
        line_total  REAL    NOT NULL,
        FOREIGN KEY (order_id)   REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );

    CREATE TABLE events (
        event_id    INTEGER PRIMARY KEY,
        session_id  INTEGER,
        customer_id INTEGER,
        event_type  TEXT,
        event_time  TEXT,
        order_id    INTEGER
    );
    """)

    # ── Load data ────────────────────────────────────────────────────────────
    customers.to_sql("customers",   conn, if_exists="append", index=False)
    products.to_sql("products",     conn, if_exists="append", index=False)
    orders.to_sql("orders",         conn, if_exists="append", index=False)
    items.to_sql("order_items",     conn, if_exists="append", index=False)
    events.to_sql("events",         conn, if_exists="append", index=False)

    # ── Indexes for fast queries ─────────────────────────────────────────────
    cur.executescript("""
    CREATE INDEX idx_orders_customer   ON orders(customer_id);
    CREATE INDEX idx_orders_date       ON orders(order_date);
    CREATE INDEX idx_orders_status     ON orders(status);
    CREATE INDEX idx_items_order       ON order_items(order_id);
    CREATE INDEX idx_items_product     ON order_items(product_id);
    CREATE INDEX idx_events_customer   ON events(customer_id);
    CREATE INDEX idx_events_type       ON events(event_type);
    CREATE INDEX idx_events_time       ON events(event_time);
    """)

    conn.commit()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n  Database summary:")
    for table in ["customers", "products", "orders", "order_items", "events"]:
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"    {table:<15} {n:>8,} rows")

    # Quick sanity check
    revenue = cur.execute(
        "SELECT ROUND(SUM(order_value),2) FROM orders WHERE status='delivered'"
    ).fetchone()[0]
    print(f"\n  Total delivered revenue: ₹{revenue:,.2f}")

    conn.close()
    print(f"\n  Done! Database saved to: {db_path.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(scale: float = 1.0):
    n_customers = max(100, int(N_CUSTOMERS * scale))
    n_products  = max(50,  int(N_PRODUCTS  * scale))
    n_orders    = max(500, int(N_ORDERS    * scale))
    n_events    = max(1000,int(200_000     * scale))

    print("=" * 55)
    print("  E-commerce Database Generator")
    print(f"  Scale: {scale:.0%}  |  Seed: {SEED}")
    print("=" * 55)
    print("\nGenerating tables...")

    customers = generate_customers(n_customers)
    products  = generate_products(n_products)
    orders, items = generate_orders(n_orders, customers, products)
    events    = generate_events(orders, customers, n_events)

    write_to_sqlite(customers, products, orders, items, events, DB_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate e-commerce SQLite database")
    parser.add_argument(
        "--rows", type=int, default=None,
        help="Target number of orders (default: 50000). Use 1000 for quick CI runs."
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Scale factor 0.0–1.0 applied to all table sizes (overridden by --rows)"
    )
    args = parser.parse_args()

    if args.rows is not None:
        scale = args.rows / N_ORDERS
    else:
        scale = args.scale

    main(scale=min(1.0, max(0.01, scale)))
