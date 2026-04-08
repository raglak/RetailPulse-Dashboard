"""
src/db.py
=========
Single file for all database access in the project.
Copy this entire file into: ecommerce-analytics/src/db.py
"""

import sqlite3
from pathlib import Path
import pandas as pd

# ── This always finds ecommerce.db no matter where you run from ──────────────
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ecommerce.db"


def _conn():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"\n  Cannot find database at: {DB_PATH}"
            f"\n  Fix: run 'python generate_data.py' from your project root first."
        )
    return sqlite3.connect(DB_PATH)


def query(sql: str) -> pd.DataFrame:
    """Run any raw SQL and return a DataFrame. Use this in notebooks to experiment."""
    with _conn() as conn:
        return pd.read_sql_query(sql, conn)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 1 — Monthly Revenue with MoM & YoY Growth
# ══════════════════════════════════════════════════════════════════════════════

def get_monthly_revenue(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Returns monthly revenue with growth metrics.
    Columns: month, year, revenue, total_orders, unique_customers,
             avg_order_value, mom_growth_pct, yoy_growth_pct,
             rolling_3m_avg, ytd_revenue, revenue_rank_in_year
    """
    date_filter = ""
    if start_date:
        date_filter += f" AND order_date >= '{start_date}'"
    if end_date:
        date_filter += f" AND order_date <= '{end_date}'"

    sql = f"""
    WITH monthly_base AS (
        SELECT
            strftime('%Y-%m', order_date)            AS month,
            CAST(strftime('%Y', order_date) AS INT)  AS year,
            CAST(strftime('%m', order_date) AS INT)  AS month_num,
            SUM(order_value)                         AS revenue,
            COUNT(DISTINCT order_id)                 AS total_orders,
            COUNT(DISTINCT customer_id)              AS unique_customers,
            ROUND(SUM(order_value) / COUNT(DISTINCT order_id), 2) AS avg_order_value
        FROM orders
        WHERE status = 'delivered' {date_filter}
        GROUP BY 1, 2, 3
    )
    SELECT
        month, year, month_num,
        ROUND(revenue, 2) AS revenue,
        total_orders,
        unique_customers,
        avg_order_value,

        ROUND(
            100.0 * (revenue - LAG(revenue, 1) OVER (ORDER BY month))
                  / NULLIF(LAG(revenue, 1) OVER (ORDER BY month), 0),
        2) AS mom_growth_pct,

        ROUND(
            100.0 * (revenue - LAG(revenue, 12) OVER (ORDER BY month))
                  / NULLIF(LAG(revenue, 12) OVER (ORDER BY month), 0),
        2) AS yoy_growth_pct,

        ROUND(
            AVG(revenue) OVER (
                ORDER BY month
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ),
        2) AS rolling_3m_avg,

        ROUND(
            SUM(revenue) OVER (
                PARTITION BY year
                ORDER BY month
                ROWS UNBOUNDED PRECEDING
            ),
        2) AS ytd_revenue,

        RANK() OVER (
            PARTITION BY year ORDER BY revenue DESC
        ) AS revenue_rank_in_year

    FROM monthly_base
    ORDER BY month;
    """
    with _conn() as conn:
        return pd.read_sql_query(sql, conn)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 2 — Cohort Retention Matrix
# ══════════════════════════════════════════════════════════════════════════════

def get_cohort_retention() -> pd.DataFrame:
    """
    Returns long-form cohort retention data.
    Columns: cohort_month, cohort_size, period_number,
             active_customers, retention_rate_pct
    """
    sql = """
    WITH customer_cohorts AS (
        SELECT DISTINCT
            customer_id,
            strftime('%Y-%m',
                MIN(order_date) OVER (PARTITION BY customer_id)
            ) AS cohort_month
        FROM orders
        WHERE status IN ('delivered', 'returned')
    ),
    order_periods AS (
        SELECT
            o.customer_id,
            c.cohort_month,
            (
                (CAST(strftime('%Y', o.order_date) AS INT) -
                 CAST(strftime('%Y', c.cohort_month || '-01') AS INT)) * 12
              + (CAST(strftime('%m', o.order_date) AS INT) -
                 CAST(strftime('%m', c.cohort_month || '-01') AS INT))
            ) AS period_number
        FROM orders o
        JOIN customer_cohorts c USING (customer_id)
        WHERE o.status IN ('delivered', 'returned')
    ),
    cohort_counts AS (
        SELECT
            cohort_month,
            period_number,
            COUNT(DISTINCT customer_id) AS active_customers
        FROM order_periods
        GROUP BY 1, 2
    ),
    cohort_sizes AS (
        SELECT cohort_month, active_customers AS cohort_size
        FROM cohort_counts
        WHERE period_number = 0
    )
    SELECT
        cc.cohort_month,
        cs.cohort_size,
        cc.period_number,
        cc.active_customers,
        ROUND(
            100.0 * cc.active_customers / NULLIF(cs.cohort_size, 0),
        1) AS retention_rate_pct
    FROM cohort_counts cc
    JOIN cohort_sizes  cs USING (cohort_month)
    WHERE cc.period_number <= 11
    ORDER BY cc.cohort_month, cc.period_number;
    """
    with _conn() as conn:
        return pd.read_sql_query(sql, conn)


def get_cohort_pivot() -> pd.DataFrame:
    """Returns cohort retention already pivoted — pass directly to px.imshow()."""
    df = get_cohort_retention()
    return df.pivot(
        index="cohort_month",
        columns="period_number",
        values="retention_rate_pct"
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 3 — RFM Customer Segmentation
# ══════════════════════════════════════════════════════════════════════════════

def get_rfm_segments() -> pd.DataFrame:
    """
    Returns one row per customer with RFM scores and segment label.
    Columns: customer_id, city, city_tier, channel, recency_days,
             frequency, monetary, r_score, f_score, m_score,
             rfm_total, rfm_segment, monetary_percentile
    """
    sql = """
    WITH snapshot AS (
        SELECT MAX(order_date) AS snapshot_date FROM orders
    ),
    rfm_raw AS (
        SELECT
            o.customer_id,
            c.city, c.city_tier, c.channel, c.age, c.gender,
            CAST(
                julianday((SELECT snapshot_date FROM snapshot)) -
                julianday(MAX(o.order_date))
            AS INT)                             AS recency_days,
            COUNT(DISTINCT o.order_id)          AS frequency,
            ROUND(SUM(o.order_value), 2)        AS monetary,
            ROUND(AVG(o.order_value), 2)        AS avg_order_value,
            MIN(o.order_date)                   AS first_order_date,
            MAX(o.order_date)                   AS last_order_date
        FROM orders o
        JOIN customers c USING (customer_id)
        WHERE o.status = 'delivered'
        GROUP BY o.customer_id, c.city, c.city_tier, c.channel, c.age, c.gender
    ),
    rfm_scored AS (
        SELECT *,
            6 - NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,
            NTILE(5)     OVER (ORDER BY frequency    ASC)  AS f_score,
            NTILE(5)     OVER (ORDER BY monetary     ASC)  AS m_score,
            ROUND(PERCENT_RANK() OVER (ORDER BY monetary ASC) * 100, 1) AS monetary_percentile
        FROM rfm_raw
    ),
    rfm_segmented AS (
        SELECT *,
            r_score + f_score + m_score AS rfm_total,
            CASE
                WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champion'
                WHEN r_score >= 3 AND f_score >= 3                  THEN 'Loyal'
                WHEN r_score >= 4 AND f_score <= 2                  THEN 'Potential'
                WHEN r_score <= 2 AND f_score >= 3                  THEN 'At-Risk'
                WHEN r_score <= 2 AND m_score >= 4                  THEN 'Cannot Lose'
                ELSE                                                      'Lost'
            END AS rfm_segment
        FROM rfm_scored
    )
    SELECT
        customer_id, city, city_tier, channel, age, gender,
        recency_days, frequency, monetary, avg_order_value,
        first_order_date, last_order_date,
        r_score, f_score, m_score, rfm_total, rfm_segment,
        monetary_percentile
    FROM rfm_segmented
    ORDER BY rfm_total DESC;
    """
    with _conn() as conn:
        return pd.read_sql_query(sql, conn)


def get_rfm_summary() -> pd.DataFrame:
    """Returns segment-level summary: count, avg spend, avg recency per segment."""
    df = get_rfm_segments()
    return (
        df.groupby("rfm_segment")
          .agg(
              customers        = ("customer_id", "count"),
              avg_spend        = ("monetary",     "mean"),
              avg_recency_days = ("recency_days", "mean"),
              avg_orders       = ("frequency",    "mean"),
              total_revenue    = ("monetary",      "sum"),
          )
          .round(2)
          .sort_values("total_revenue", ascending=False)
          .reset_index()
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 4 — Product Pareto (80/20 Cumulative Revenue)
# ══════════════════════════════════════════════════════════════════════════════

def get_product_pareto(category: str = None) -> pd.DataFrame:
    """
    Returns all products ranked by revenue with cumulative Pareto % column.
    Columns: product_id, product_name, category, revenue, revenue_rank,
             revenue_share_pct, cumulative_revenue_pct, is_top_80pct_driver

    Args:
        category: optional filter e.g. 'Electronics'. None = all categories.
    """
    cat_filter = f"AND p.category = '{category}'" if category else ""

    sql = f"""
    WITH product_revenue AS (
        SELECT
            p.product_id,
            p.name        AS product_name,
            p.category,
            p.price,
            p.margin_pct,
            p.rating,
            COUNT(DISTINCT o.order_id)           AS total_orders,
            SUM(oi.quantity)                     AS units_sold,
            ROUND(SUM(oi.line_total), 2)         AS revenue,
            ROUND(SUM(oi.line_total)
                  - SUM(oi.quantity * p.cost),2) AS gross_profit,
            COUNT(DISTINCT o.customer_id)        AS unique_buyers,
            ROUND(AVG(oi.discount), 2)           AS avg_discount
        FROM products    p
        JOIN order_items oi ON oi.product_id = p.product_id
        JOIN orders      o  ON o.order_id    = oi.order_id
        WHERE o.status = 'delivered' {cat_filter}
        GROUP BY p.product_id, p.name, p.category, p.price, p.margin_pct, p.rating
    ),
    totals AS (
        SELECT SUM(revenue) AS grand_revenue FROM product_revenue
    ),
    pareto AS (
        SELECT
            pr.*,
            RANK() OVER (ORDER BY pr.revenue DESC) AS revenue_rank,
            ROUND(100.0 * pr.revenue / NULLIF(t.grand_revenue,0), 2) AS revenue_share_pct,
            ROUND(
                100.0 * SUM(pr.revenue) OVER (
                    ORDER BY pr.revenue DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) / NULLIF(t.grand_revenue, 0),
            2) AS cumulative_revenue_pct,
            ROUND(100.0 * pr.gross_profit / NULLIF(pr.revenue,0), 1) AS realized_margin_pct,
            NTILE(4) OVER (ORDER BY pr.revenue DESC) AS revenue_quartile
        FROM product_revenue pr
        CROSS JOIN totals t
    )
    SELECT *,
        CASE WHEN cumulative_revenue_pct <= 80 THEN 1 ELSE 0 END AS is_top_80pct_driver
    FROM pareto
    ORDER BY revenue_rank;
    """
    with _conn() as conn:
        return pd.read_sql_query(sql, conn)


def get_category_revenue() -> pd.DataFrame:
    """Returns revenue breakdown and share % by product category."""
    sql = """
    SELECT
        p.category,
        COUNT(DISTINCT p.product_id)  AS products,
        COUNT(DISTINCT o.order_id)    AS orders,
        SUM(oi.quantity)              AS units_sold,
        ROUND(SUM(oi.line_total), 2)  AS revenue,
        ROUND(AVG(p.margin_pct), 1)   AS avg_margin_pct,
        ROUND(
            100.0 * SUM(oi.line_total)
                  / SUM(SUM(oi.line_total)) OVER (),
        2) AS revenue_share_pct,
        RANK() OVER (ORDER BY SUM(oi.line_total) DESC) AS revenue_rank
    FROM products    p
    JOIN order_items oi ON oi.product_id = p.product_id
    JOIN orders      o  ON o.order_id    = oi.order_id
    WHERE o.status = 'delivered'
    GROUP BY p.category
    ORDER BY revenue DESC;
    """
    with _conn() as conn:
        return pd.read_sql_query(sql, conn)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 5 — Funnel Conversion Rates
# ══════════════════════════════════════════════════════════════════════════════

def get_funnel_conversion() -> pd.DataFrame:
    """
    Returns overall funnel: view → add_to_cart → checkout → purchase.
    Columns: stage, sessions, step_conversion_pct, overall_conversion_pct,
             sessions_dropped, dropoff_pct, estimated_lost_revenue
    """
    sql = """
    WITH stage_sessions AS (
        SELECT
            event_type,
            COUNT(DISTINCT session_id) AS sessions,
            CASE event_type
                WHEN 'view'        THEN 1
                WHEN 'add_to_cart' THEN 2
                WHEN 'checkout'    THEN 3
                WHEN 'purchase'    THEN 4
            END AS stage_order
        FROM events
        WHERE event_type IN ('view','add_to_cart','checkout','purchase')
        GROUP BY event_type
    ),
    with_conversions AS (
        SELECT
            stage_order,
            event_type AS stage,
            sessions,
            LAG(sessions) OVER (ORDER BY stage_order)         AS prev_stage_sessions,
            FIRST_VALUE(sessions) OVER (ORDER BY stage_order) AS top_of_funnel,
            ROUND(
                100.0 * sessions
                      / NULLIF(LAG(sessions) OVER (ORDER BY stage_order), 0),
            1) AS step_conversion_pct,
            ROUND(
                100.0 * sessions
                      / NULLIF(FIRST_VALUE(sessions) OVER (ORDER BY stage_order), 0),
            1) AS overall_conversion_pct,
            LAG(sessions) OVER (ORDER BY stage_order) - sessions AS sessions_dropped,
            ROUND(
                100.0 * (LAG(sessions) OVER (ORDER BY stage_order) - sessions)
                      / NULLIF(LAG(sessions) OVER (ORDER BY stage_order), 0),
            1) AS dropoff_pct
        FROM stage_sessions
    )
    SELECT
        stage_order, stage, sessions,
        step_conversion_pct, overall_conversion_pct,
        sessions_dropped, dropoff_pct,
        ROUND(
            COALESCE(sessions_dropped, 0) *
            (SELECT AVG(order_value) FROM orders WHERE status = 'delivered'),
        2) AS estimated_lost_revenue
    FROM with_conversions
    ORDER BY stage_order;
    """
    with _conn() as conn:
        return pd.read_sql_query(sql, conn)


def get_funnel_by_segment() -> pd.DataFrame:
    """Funnel conversion broken down by acquisition channel and city tier."""
    sql = """
    WITH stage_counts AS (
        SELECT
            c.channel,
            c.city_tier,
            e.event_type,
            CASE e.event_type
                WHEN 'view'        THEN 1
                WHEN 'add_to_cart' THEN 2
                WHEN 'checkout'    THEN 3
                WHEN 'purchase'    THEN 4
            END AS stage_order,
            COUNT(DISTINCT e.session_id) AS sessions
        FROM events    e
        JOIN customers c ON c.customer_id = e.customer_id
        WHERE e.event_type IN ('view','add_to_cart','checkout','purchase')
        GROUP BY c.channel, c.city_tier, e.event_type
    ),
    with_totals AS (
        SELECT *,
            FIRST_VALUE(sessions) OVER (
                PARTITION BY channel, city_tier
                ORDER BY stage_order
            ) AS top_of_funnel
        FROM stage_counts
    )
    SELECT
        channel, city_tier,
        event_type AS stage,
        stage_order, sessions,
        ROUND(100.0 * sessions / NULLIF(top_of_funnel,0), 1) AS overall_conv_pct
    FROM with_totals
    ORDER BY channel, city_tier, stage_order;
    """
    with _conn() as conn:
        return pd.read_sql_query(sql, conn)
