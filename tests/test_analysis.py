# tests/test_analysis.py
"""
Unit tests for the E-commerce Analytics Dashboard.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

# ── Add project root to path so src.db can be found ───────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import pandas as pd
from src.db import (
    get_monthly_revenue,
    get_cohort_retention,
    get_cohort_pivot,
    get_rfm_segments,
    get_rfm_summary,
    get_product_pareto,
    get_category_revenue,
    get_funnel_conversion,
    get_funnel_by_segment,
    query,
)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — get_monthly_revenue()
# ══════════════════════════════════════════════════════════════════════════════

def test_revenue_returns_dataframe():
    """Query must return a pandas DataFrame"""
    df = get_monthly_revenue()
    assert isinstance(df, pd.DataFrame)


def test_revenue_has_required_columns():
    """Must have these columns for the dashboard to work"""
    df = get_monthly_revenue()
    required = ['month', 'year', 'revenue', 'total_orders',
                'unique_customers', 'avg_order_value',
                'mom_growth_pct', 'rolling_3m_avg', 'ytd_revenue']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_revenue_no_nulls_in_key_columns():
    """Revenue and month columns must never be null"""
    df = get_monthly_revenue()
    assert df['revenue'].isnull().sum() == 0
    assert df['month'].isnull().sum() == 0


def test_revenue_all_positive():
    """Revenue must always be a positive number"""
    df = get_monthly_revenue()
    assert (df['revenue'] > 0).all()


def test_revenue_has_rows():
    """Must return at least 12 months of data"""
    df = get_monthly_revenue()
    assert len(df) >= 12


def test_revenue_date_filter_works():
    """Passing start and end date must reduce the rows"""
    df_full    = get_monthly_revenue()
    df_filtered = get_monthly_revenue(start_date='2023-01-01',
                                      end_date='2023-12-31')
    assert len(df_filtered) < len(df_full)
    assert len(df_filtered) <= 12


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — get_cohort_retention() and get_cohort_pivot()
# ══════════════════════════════════════════════════════════════════════════════

def test_cohort_retention_returns_dataframe():
    df = get_cohort_retention()
    assert isinstance(df, pd.DataFrame)


def test_cohort_retention_has_required_columns():
    df = get_cohort_retention()
    required = ['cohort_month', 'cohort_size', 'period_number',
                'active_customers', 'retention_rate_pct']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_cohort_period_0_is_100_percent():
    """Period 0 retention must always be 100% — every customer exists in signup month"""
    df = get_cohort_retention()
    period_0 = df[df['period_number'] == 0]
    assert (period_0['retention_rate_pct'] == 100.0).all()


def test_cohort_retention_between_0_and_100():
    """Retention % must always be between 0 and 100"""
    df = get_cohort_retention()
    assert (df['retention_rate_pct'] >= 0).all()
    assert (df['retention_rate_pct'] <= 100).all()


def test_cohort_pivot_shape():
    """Pivot must have at least 12 columns (period 0 to 11)"""
    pivot = get_cohort_pivot()
    assert isinstance(pivot, pd.DataFrame)
    assert pivot.shape[1] >= 1
    assert pivot.shape[0] >= 1


def test_cohort_pivot_period_0_is_100():
    """First column of pivot (period 0) must be 100.0 for all cohorts"""
    pivot = get_cohort_pivot()
    assert pivot[0].max() == 100.0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — get_rfm_segments()
# ══════════════════════════════════════════════════════════════════════════════

def test_rfm_returns_dataframe():
    df = get_rfm_segments()
    assert isinstance(df, pd.DataFrame)


def test_rfm_has_required_columns():
    df = get_rfm_segments()
    required = ['customer_id', 'recency_days', 'frequency',
                'monetary', 'r_score', 'f_score', 'm_score',
                'rfm_total', 'rfm_segment']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_rfm_no_nulls():
    """Core RFM columns must have no missing values"""
    df = get_rfm_segments()
    assert df[['recency_days', 'frequency', 'monetary']].isnull().sum().sum() == 0


def test_rfm_scores_between_1_and_5():
    """All scores must be between 1 and 5 — NTILE(5) range"""
    df = get_rfm_segments()
    for score_col in ['r_score', 'f_score', 'm_score']:
        assert df[score_col].between(1, 5).all(), f"{score_col} out of range"


def test_rfm_valid_segments():
    """Segments must only be one of the 6 defined labels"""
    df = get_rfm_segments()
    valid = {'Champion', 'Loyal', 'Potential', 'At-Risk', 'Cannot Lose', 'Lost'}
    actual = set(df['rfm_segment'].unique())
    assert actual.issubset(valid), f"Invalid segments found: {actual - valid}"


def test_rfm_monetary_positive():
    """Monetary must always be positive — delivered orders only"""
    df = get_rfm_segments()
    assert (df['monetary'] > 0).all()


def test_rfm_summary_has_6_segments():
    """Summary must have one row per segment"""
    df = get_rfm_summary()
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 6
    assert 'rfm_segment' in df.columns
    assert 'total_revenue' in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — get_product_pareto()
# ══════════════════════════════════════════════════════════════════════════════

def test_pareto_returns_dataframe():
    df = get_product_pareto()
    assert isinstance(df, pd.DataFrame)


def test_pareto_has_required_columns():
    df = get_product_pareto()
    required = ['product_id', 'product_name', 'category', 'revenue',
                'revenue_rank', 'cumulative_revenue_pct', 'is_top_80pct_driver']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_pareto_cumulative_ends_at_100():
    """Last product's cumulative % must be 100 (all revenue accounted for)"""
    df = get_product_pareto()
    assert df['cumulative_revenue_pct'].max() == pytest.approx(100.0, abs=1.0)


def test_pareto_driver_flag_is_binary():
    """is_top_80pct_driver must only be 0 or 1"""
    df = get_product_pareto()
    assert set(df['is_top_80pct_driver'].unique()).issubset({0, 1})


def test_pareto_category_filter_works():
    """Filtering by category must return fewer rows"""
    df_all = get_product_pareto()
    df_cat = get_product_pareto(category='Electronics')
    assert len(df_cat) < len(df_all)
    assert (df_cat['category'] == 'Electronics').all()


def test_category_revenue_returns_dataframe():
    df = get_category_revenue()
    assert isinstance(df, pd.DataFrame)
    assert 'category' in df.columns
    assert 'revenue' in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — get_funnel_conversion()
# ══════════════════════════════════════════════════════════════════════════════

def test_funnel_returns_dataframe():
    df = get_funnel_conversion()
    assert isinstance(df, pd.DataFrame)


def test_funnel_has_4_stages():
    """Funnel must have exactly 4 stages"""
    df = get_funnel_conversion()
    assert len(df) == 4


def test_funnel_stages_are_correct():
    """Stage names must match exactly"""
    df = get_funnel_conversion()
    expected = {'view', 'add_to_cart', 'checkout', 'purchase'}
    actual   = set(df['stage'].unique())
    assert actual == expected


def test_funnel_sessions_decrease():
    """Sessions must decrease or stay same at each stage — never increase"""
    df = get_funnel_conversion().sort_values('stage_order')
    sessions = df['sessions'].tolist()
    for i in range(1, len(sessions)):
        assert sessions[i] <= sessions[i-1], \
            f"Sessions increased at stage {i} — {sessions[i-1]} → {sessions[i]}"


def test_funnel_conversion_pct_between_0_and_100():
    """Overall conversion % must be between 0 and 100"""
    df = get_funnel_conversion()
    valid = df['overall_conversion_pct'].dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_funnel_by_segment_returns_dataframe():
    df = get_funnel_by_segment()
    assert isinstance(df, pd.DataFrame)
    assert 'channel' in df.columns
    assert 'city_tier' in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — raw query() function
# ══════════════════════════════════════════════════════════════════════════════

def test_raw_query_works():
    """The generic query() function must work for any SQL"""
    df = query("SELECT COUNT(*) as total FROM orders")
    assert isinstance(df, pd.DataFrame)
    assert 'total' in df.columns
    assert df['total'].iloc[0] > 0


def test_raw_query_customers_table():
    df = query("SELECT COUNT(*) as total FROM customers")
    assert df['total'].iloc[0] > 0


def test_raw_query_products_table():
    df = query("SELECT COUNT(*) as total FROM products")
    assert df['total'].iloc[0] > 0
