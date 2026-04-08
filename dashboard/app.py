# dashboard/app.py
import sys
from pathlib import Path

# ── Fix: add project root to path so src.db can be found ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd

from src.db import (
    get_monthly_revenue,
    get_cohort_pivot,
    get_rfm_segments,
    get_rfm_summary,
    get_product_pareto,
    get_category_revenue,
    get_funnel_conversion,
)

# ── App init ───────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # needed for deployment on Render

# ── Load data once at startup ──────────────────────────────────────────────────
revenue_df  = get_monthly_revenue()
cohort_df   = get_cohort_pivot()
rfm_df      = get_rfm_segments()
rfm_sum_df  = get_rfm_summary()
pareto_df   = get_product_pareto()
cat_df      = get_category_revenue()
funnel_df   = get_funnel_conversion()

# ── KPI values ─────────────────────────────────────────────────────────────────
total_revenue    = revenue_df['revenue'].sum()
total_customers  = rfm_df['customer_id'].nunique()
avg_order_value  = revenue_df['avg_order_value'].mean()
churn_rate       = (rfm_df['recency_days'] >= 90).mean() * 100

# ── Helper: KPI card ───────────────────────────────────────────────────────────
def kpi_card(title, value):
    return html.Div([
        html.P(title, style={'fontSize':'13px','color':'#888','margin':'0'}),
        html.H3(value, style={'fontSize':'22px','fontWeight':'500','margin':'4px 0 0'}),
    ], style={
        'background':'#f8f8f8','borderRadius':'10px',
        'padding':'16px 20px','flex':'1','minWidth':'140px'
    })

# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.Div([
        html.H1("E-commerce Analytics Dashboard",
                style={'fontSize':'22px','fontWeight':'500','margin':'0'}),
        html.P("3 years of sales data — Jan 2022 to Dec 2024",
               style={'fontSize':'13px','color':'#888','margin':'4px 0 0'}),
    ], style={'padding':'20px 24px','borderBottom':'1px solid #eee'}),

    # KPI row
    html.Div([
        kpi_card("Total Revenue",      f"₹{total_revenue:,.0f}"),
        kpi_card("Total Customers",    f"{total_customers:,}"),
        kpi_card("Avg Order Value",    f"₹{avg_order_value:,.0f}"),
        kpi_card("Churn Rate",         f"{churn_rate:.1f}%"),
    ], style={'display':'flex','gap':'16px','padding':'20px 24px'}),

    # Date filter
    html.Div([
        html.Label("Filter by date range:",
                   style={'fontSize':'13px','color':'#555','marginRight':'10px'}),
        dcc.DatePickerRange(
            id='date-range',
            start_date='2022-01-01',
            end_date='2024-12-31',
            display_format='YYYY-MM-DD',
        ),
    ], style={'padding':'0 24px 16px','display':'flex','alignItems':'center'}),

    # Tabs
    dcc.Tabs(id='tabs', value='revenue',
             style={'padding':'0 24px'},
             children=[
                 dcc.Tab(label='Revenue',  value='revenue'),
                 dcc.Tab(label='Cohorts',  value='cohorts'),
                 dcc.Tab(label='RFM',      value='rfm'),
                 dcc.Tab(label='Products', value='products'),
                 dcc.Tab(label='Funnel',   value='funnel'),
             ]),

    # Tab content area
    html.Div(id='tab-content', style={'padding':'24px'}),

], style={'fontFamily':'sans-serif','maxWidth':'1200px','margin':'0 auto'})


# ── Callback: render tab content ───────────────────────────────────────────────
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
)
def render_tab(tab, start, end):

    # ── TAB 1: Revenue ─────────────────────────────────────────────────────────
    if tab == 'revenue':
        df = get_monthly_revenue(start, end)

        fig1 = px.line(df, x='month', y='revenue',
                       title='Monthly revenue',
                       labels={'revenue':'Revenue (₹)','month':'Month'})
        fig1.add_scatter(x=df['month'], y=df['rolling_3m_avg'],
                         name='3-month avg', line=dict(dash='dot', color='orange'))

        fig2 = px.bar(df, x='month', y='mom_growth_pct',
                      title='Month-over-Month growth %',
                      labels={'mom_growth_pct':'MoM Growth %','month':'Month'},
                      color='mom_growth_pct',
                      color_continuous_scale='RdYlGn')

        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
        ])

    # ── TAB 2: Cohorts ─────────────────────────────────────────────────────────
    elif tab == 'cohorts':
        fig = px.imshow(
            cohort_df,
            color_continuous_scale='RdYlGn',
            text_auto='.0f',
            title='Cohort retention rate (%)',
            labels=dict(x='Month number', y='Cohort', color='Retention %'),
            aspect='auto',
        )
        fig.update_layout(height=600)

        return html.Div([
            html.P("Each row = a group of customers who first bought in that month. "
                   "Each column = how many months later. Values = % still active.",
                   style={'fontSize':'13px','color':'#666','marginBottom':'12px'}),
            dcc.Graph(figure=fig),
        ])

    # ── TAB 3: RFM ─────────────────────────────────────────────────────────────
    elif tab == 'rfm':
        fig1 = px.bar(rfm_sum_df, x='rfm_segment', y='customers',
                      color='rfm_segment',
                      title='Customers per segment',
                      labels={'customers':'Customers','rfm_segment':'Segment'})

        fig2 = px.bar(rfm_sum_df, x='rfm_segment', y='total_revenue',
                      color='rfm_segment',
                      title='Revenue per segment',
                      labels={'total_revenue':'Revenue (₹)','rfm_segment':'Segment'})

        fig3 = px.scatter(rfm_df.sample(min(2000, len(rfm_df))),
                          x='recency_days', y='monetary',
                          color='rfm_segment',
                          title='Customer map — recency vs spend',
                          labels={'recency_days':'Days since last purchase',
                                  'monetary':'Total spend (₹)'},
                          hover_data=['customer_id','city','frequency'],
                          opacity=0.6)

        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3),
        ])

    # ── TAB 4: Products ────────────────────────────────────────────────────────
    elif tab == 'products':
        top30 = pareto_df.head(30)

        fig1 = px.bar(top30, x='product_name', y='revenue',
                      color='is_top_80pct_driver',
                      color_discrete_map={1:'#1D9E75', 0:'#B4B2A9'},
                      title='Top 30 products — green drives 80% of revenue',
                      labels={'revenue':'Revenue (₹)','product_name':'Product'})
        fig1.update_layout(xaxis_tickangle=-45)

        fig2 = px.bar(cat_df, x='category', y='revenue',
                      color='revenue_share_pct',
                      title='Revenue by category',
                      labels={'revenue':'Revenue (₹)','category':'Category'},
                      text='revenue_share_pct',
                      color_continuous_scale='Blues')

        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
        ])

    # ── TAB 5: Funnel ──────────────────────────────────────────────────────────
    elif tab == 'funnel':
        fig = px.funnel(funnel_df, x='sessions', y='stage',
                        title='Customer purchase funnel',
                        labels={'sessions':'Sessions','stage':'Stage'})

        # Drop-off table
        table = dash_table.DataTable(
            data=funnel_df[['stage','sessions','step_conversion_pct',
                            'dropoff_pct','estimated_lost_revenue']].to_dict('records'),
            columns=[
                {'name':'Stage',              'id':'stage'},
                {'name':'Sessions',           'id':'sessions'},
                {'name':'Step conversion %',  'id':'step_conversion_pct'},
                {'name':'Drop-off %',         'id':'dropoff_pct'},
                {'name':'Lost revenue (₹)',   'id':'estimated_lost_revenue'},
            ],
            style_table={'overflowX':'auto'},
            style_cell={'textAlign':'left','padding':'8px','fontSize':'13px'},
            style_header={'fontWeight':'bold','backgroundColor':'#f0f0f0'},
        )

        return html.Div([
            dcc.Graph(figure=fig),
            html.H4("Drop-off detail",
                    style={'marginTop':'24px','fontSize':'15px','fontWeight':'500'}),
            table,
        ])


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
