# dashboard/app.py
import sys
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import anthropic
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import plotly.io as pio
import io

# ── Fix: add project root to path so src.db can be found ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
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

# ── Anthropic client ───────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SCHEMA = """
tables: orders(order_id, customer_id, product_id, order_date, quantity, total_amount, status, channel, city),
customers(customer_id, name, city, city_tier, signup_date),
products(product_id, product_name, category, price)
"""

def nl_to_sql(question: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"Convert to SQL. Schema: {SCHEMA}\nQuestion: {question}\nReturn only the SQL query, no explanation."
        }]
    )
    return response.content[0].text.strip()

# ── Color Palette ──────────────────────────────────────────────────────────────
DARK_BG     = '#0D0F14'
CARD_BG     = '#151821'
CARD_BORDER = '#1E2330'
ACCENT      = '#00E5A0'
ACCENT2     = '#6C63FF'
ACCENT3     = '#FF6B6B'
TEXT_PRI    = '#F0F4FF'
TEXT_SEC    = '#8B92A5'
GRID_COLOR  = '#1E2330'

# ── Plotly theme ───────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color=TEXT_SEC, size=12),
    title_font=dict(family='DM Sans, sans-serif', color=TEXT_PRI, size=15),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=CARD_BORDER, tickfont=dict(color=TEXT_SEC)),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=CARD_BORDER, tickfont=dict(color=TEXT_SEC)),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_SEC)),
    coloraxis_colorbar=dict(tickfont=dict(color=TEXT_SEC), title_font=dict(color=TEXT_SEC)),
)

# ── Inline CSS ─────────────────────────────────────────────────────────────────
EXTERNAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: #0D0F14;
    font-family: 'DM Sans', sans-serif;
    color: #F0F4FF;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D0F14; }
::-webkit-scrollbar-thumb { background: #1E2330; border-radius: 3px; }

.tab-container .tab {
    background: transparent !important;
    border: none !important;
    color: #8B92A5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.tab-container .tab--selected {
    background: rgba(0,229,160,0.1) !important;
    color: #00E5A0 !important;
    border-bottom: 2px solid #00E5A0 !important;
}
.tab-container .tab:hover {
    color: #F0F4FF !important;
    background: rgba(255,255,255,0.05) !important;
}

.DateInput_input {
    background: #151821 !important;
    color: #F0F4FF !important;
    border: 1px solid #1E2330 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    border-radius: 8px !important;
}
.DateRangePickerInput {
    background: #151821 !important;
    border: 1px solid #1E2330 !important;
    border-radius: 8px !important;
}
.DateRangePickerInput_arrow { color: #8B92A5 !important; }

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
    background: #151821 !important;
    color: #F0F4FF !important;
    border-color: #1E2330 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
    background: #1E2330 !important;
    color: #00E5A0 !important;
    border-color: #1E2330 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

input[type=text] {
    background: #151821 !important;
    color: #F0F4FF !important;
    border: 1px solid #1E2330 !important;
    font-family: 'DM Sans', sans-serif !important;
}
input[type=text]:focus {
    outline: none !important;
    border-color: #00E5A0 !important;
    box-shadow: 0 0 0 3px rgba(0,229,160,0.1) !important;
}
input[type=text]::placeholder { color: #8B92A5 !important; }
"""

# ── App init ───────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>E-commerce Analytics</title>
        {%favicon%}
        {%css%}
        <style>''' + EXTERNAL_CSS + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

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
def kpi_card(title, value, icon, delta=None, delta_positive=True):
    delta_color = ACCENT if delta_positive else ACCENT3
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize':'20px'}),
        ], style={
            'width':'40px','height':'40px','borderRadius':'10px',
            'background':'rgba(0,229,160,0.1)',
            'display':'flex','alignItems':'center','justifyContent':'center',
            'marginBottom':'12px'
        }),
        html.P(title, style={
            'fontSize':'11px','color':TEXT_SEC,'margin':'0',
            'textTransform':'uppercase','letterSpacing':'0.08em','fontWeight':'600'
        }),
        html.H3(value, style={
            'fontSize':'24px','fontWeight':'700','margin':'4px 0 0',
            'color':TEXT_PRI,'letterSpacing':'-0.02em'
        }),
        html.P(delta, style={
            'fontSize':'12px','color':delta_color,'margin':'6px 0 0','fontWeight':'500'
        }) if delta else html.Div(),
    ], style={
        'background':CARD_BG,
        'border':f'1px solid {CARD_BORDER}',
        'borderRadius':'16px',
        'padding':'20px',
        'flex':'1','minWidth':'160px',
    })

# ── Chart card wrapper ─────────────────────────────────────────────────────────
def chart_card(children):
    return html.Div(children, style={
        'background':CARD_BG,
        'border':f'1px solid {CARD_BORDER}',
        'borderRadius':'16px',
        'padding':'8px',
        'marginBottom':'16px',
    })

# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Sidebar strip
    html.Div(style={
        'position':'fixed','left':'0','top':'0','bottom':'0','width':'3px',
        'background':f'linear-gradient(180deg, {ACCENT}, {ACCENT2}, {ACCENT3})',
        'zIndex':'999'
    }),

    html.Div([

        # Header
        html.Div([
            html.Div([
                html.Div([
                    html.Span('◈', style={'color':ACCENT,'fontSize':'20px','marginRight':'10px'}),
                    html.Span("E-commerce Analytics", style={
                        'fontSize':'18px','fontWeight':'700','color':TEXT_PRI,'letterSpacing':'-0.02em'
                    }),
                ], style={'display':'flex','alignItems':'center'}),
                html.P("3 years of sales data  ·  Jan 2022 – Dec 2024", style={
                    'fontSize':'12px','color':TEXT_SEC,'margin':'4px 0 0 30px'
                }),
            ]),
            html.Div('● LIVE', style={
                'fontSize':'11px','color':ACCENT,'fontWeight':'600','letterSpacing':'0.1em'
            }),
        ], style={
            'display':'flex','justifyContent':'space-between','alignItems':'center',
            'padding':'20px 32px','borderBottom':f'1px solid {CARD_BORDER}',
            'background':'rgba(13,15,20,0.95)',
            'position':'sticky','top':'0','zIndex':'100',
            'backdropFilter':'blur(10px)',
        }),

        # KPI row
        html.Div([
            kpi_card("Total Revenue",   f"₹{total_revenue/1e6:.1f}M",  "💰", "↑ All time"),
            kpi_card("Total Customers", f"{total_customers:,}",         "👥", "↑ Active base"),
            kpi_card("Avg Order Value", f"₹{avg_order_value:,.0f}",    "🛒", "Per transaction"),
            kpi_card("Churn Rate",      f"{churn_rate:.1f}%",           "📉", "↓ 90d inactive", False),
        ], style={'display':'flex','gap':'16px','padding':'24px 32px'}),

        # Controls row
        html.Div([
            html.Div([
                html.Label("Date range", style={
                    'fontSize':'11px','color':TEXT_SEC,'textTransform':'uppercase',
                    'letterSpacing':'0.08em','fontWeight':'600','marginBottom':'6px','display':'block'
                }),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date='2022-01-01',
                    end_date='2024-12-31',
                    display_format='YYYY-MM-DD',
                ),
            ]),
            html.Div([
                html.Button("📥 Export PDF", id='download-btn', style={
                    'padding':'10px 20px',
                    'background':f'linear-gradient(135deg, {ACCENT}, #00B880)',
                    'color':'#0D0F14','border':'none','borderRadius':'10px',
                    'cursor':'pointer','fontSize':'13px','fontWeight':'700',
                    'fontFamily':'DM Sans, sans-serif',
                    'boxShadow':'0 4px 20px rgba(0,229,160,0.3)',
                }),
                dcc.Download(id='download-pdf'),
            ]),
        ], style={
            'display':'flex','justifyContent':'space-between','alignItems':'flex-end',
            'padding':'0 32px 20px',
        }),

        # Tabs
        html.Div([
            dcc.Tabs(id='tabs', value='revenue',
                     className='tab-container',
                     style={'borderBottom':f'1px solid {CARD_BORDER}'},
                     children=[
                         dcc.Tab(label='📈  Revenue',  value='revenue'),
                         dcc.Tab(label='🔁  Cohorts',  value='cohorts'),
                         dcc.Tab(label='🎯  RFM',      value='rfm'),
                         dcc.Tab(label='📦  Products', value='products'),
                         dcc.Tab(label='🔽  Funnel',   value='funnel'),
                         dcc.Tab(label='🤖  Ask AI',   value='ask_ai'),
                     ]),
        ], style={'padding':'0 32px'}),

        # Tab content
        html.Div(id='tab-content', style={'padding':'24px 32px'}),

    ], style={'marginLeft':'3px'}),

], style={'background':DARK_BG,'minHeight':'100vh','fontFamily':'DM Sans, sans-serif'})


# ── Callback: render tab content ───────────────────────────────────────────────
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
)
def render_tab(tab, start, end):

    # TAB 1: Revenue
    if tab == 'revenue':
        df = get_monthly_revenue(start, end)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['month'], y=df['revenue'],
            name='Revenue', mode='lines',
            line=dict(color=ACCENT, width=2.5),
            fill='tozeroy', fillcolor='rgba(0,229,160,0.07)'
        ))
        fig1.add_trace(go.Scatter(
            x=df['month'], y=df['rolling_3m_avg'],
            name='3-month avg', mode='lines',
            line=dict(color=ACCENT2, width=1.5, dash='dot')
        ))
        fig1.update_layout(title='Monthly Revenue', **CHART_LAYOUT)

        fig2 = px.bar(df, x='month', y='mom_growth_pct',
                      title='Month-over-Month Growth %',
                      color='mom_growth_pct',
                      color_continuous_scale=[[0,'#FF6B6B'],[0.5,'#FFD166'],[1,'#00E5A0']])
        fig2.update_layout(**CHART_LAYOUT)

        return html.Div([
            chart_card(dcc.Graph(figure=fig1, config={'displayModeBar':False})),
            chart_card(dcc.Graph(figure=fig2, config={'displayModeBar':False})),
        ])

    # TAB 2: Cohorts
    elif tab == 'cohorts':
        fig = px.imshow(
            cohort_df,
            color_continuous_scale=[[0,'#151821'],[0.4,'#6C63FF'],[1,'#00E5A0']],
            text_auto='.0f',
            title='Cohort Retention Rate (%)',
            labels=dict(x='Month number', y='Cohort', color='Retention %'),
            aspect='auto',
        )
        fig.update_layout(height=560, **CHART_LAYOUT)
        fig.update_traces(textfont=dict(color='white', size=10))

        return html.Div([
            html.Div([
                html.Span('ℹ', style={'color':ACCENT,'marginRight':'8px','fontSize':'14px'}),
                html.Span(
                    "Each row = customers who first bought in that month. "
                    "Each column = months later. Values = % still active.",
                    style={'fontSize':'13px','color':TEXT_SEC}
                ),
            ], style={
                'background':'rgba(0,229,160,0.06)',
                'border':'1px solid rgba(0,229,160,0.15)',
                'borderRadius':'10px','padding':'12px 16px','marginBottom':'16px',
                'display':'flex','alignItems':'center'
            }),
            chart_card(dcc.Graph(figure=fig, config={'displayModeBar':False})),
        ])

    # TAB 3: RFM
    elif tab == 'rfm':
        SEGMENT_COLORS = {
            'Champion':'#00E5A0','Loyal':'#6C63FF','Potential':'#FFD166',
            'At-Risk':'#FF9F43','Cannot Lose':'#FF6B6B','Lost':'#8B92A5'
        }

        fig1 = px.bar(rfm_sum_df, x='rfm_segment', y='customers',
                      color='rfm_segment', color_discrete_map=SEGMENT_COLORS,
                      title='Customers per Segment')
        fig1.update_layout(**CHART_LAYOUT, showlegend=False)

        fig2 = px.bar(rfm_sum_df, x='rfm_segment', y='total_revenue',
                      color='rfm_segment', color_discrete_map=SEGMENT_COLORS,
                      title='Revenue per Segment')
        fig2.update_layout(**CHART_LAYOUT, showlegend=False)

        fig3 = px.scatter(rfm_df.sample(min(2000, len(rfm_df))),
                          x='recency_days', y='monetary',
                          color='rfm_segment', color_discrete_map=SEGMENT_COLORS,
                          title='Customer Map — Recency vs Spend',
                          labels={'recency_days':'Days since last purchase','monetary':'Total spend (₹)'},
                          hover_data=['customer_id','city','frequency'],
                          opacity=0.7)
        fig3.update_layout(height=420, **CHART_LAYOUT)

        return html.Div([
            html.Div([
                html.Div([chart_card(dcc.Graph(figure=fig1, config={'displayModeBar':False}))],
                         style={'flex':'1'}),
                html.Div([chart_card(dcc.Graph(figure=fig2, config={'displayModeBar':False}))],
                         style={'flex':'1'}),
            ], style={'display':'flex','gap':'16px'}),
            chart_card(dcc.Graph(figure=fig3, config={'displayModeBar':False})),
        ])

    # TAB 4: Products
    elif tab == 'products':
        top30 = pareto_df.head(30)

        fig1 = px.bar(top30, x='product_name', y='revenue',
                      color='is_top_80pct_driver',
                      color_discrete_map={1:ACCENT, 0:'#2A2F3E'},
                      title='Top 30 Products — Green = 80% Revenue Drivers',
                      labels={'revenue':'Revenue (₹)','product_name':'Product'})
        fig1.update_layout(xaxis_tickangle=-45, showlegend=False, **CHART_LAYOUT)

        fig2 = px.bar(cat_df, x='category', y='revenue',
                      color='revenue_share_pct',
                      title='Revenue by Category',
                      labels={'revenue':'Revenue (₹)','category':'Category'},
                      text='revenue_share_pct',
                      color_continuous_scale=[[0,'#1E2330'],[1,ACCENT2]])
        fig2.update_layout(**CHART_LAYOUT)

        return html.Div([
            chart_card(dcc.Graph(figure=fig1, config={'displayModeBar':False})),
            chart_card(dcc.Graph(figure=fig2, config={'displayModeBar':False})),
        ])

    # TAB 5: Funnel
    elif tab == 'funnel':
        fig = px.funnel(funnel_df, x='sessions', y='stage',
                        title='Customer Purchase Funnel',
                        color_discrete_sequence=[ACCENT, ACCENT2, '#FFD166', ACCENT3])
        fig.update_layout(**CHART_LAYOUT)

        table = dash_table.DataTable(
            data=funnel_df[['stage','sessions','step_conversion_pct',
                            'dropoff_pct','estimated_lost_revenue']].to_dict('records'),
            columns=[
                {'name':'Stage',            'id':'stage'},
                {'name':'Sessions',         'id':'sessions'},
                {'name':'Step Conv. %',     'id':'step_conversion_pct'},
                {'name':'Drop-off %',       'id':'dropoff_pct'},
                {'name':'Lost Revenue (₹)', 'id':'estimated_lost_revenue'},
            ],
            style_table={'overflowX':'auto','borderRadius':'12px','overflow':'hidden'},
            style_cell={'textAlign':'left','padding':'12px 16px','border':'none',
                        'fontFamily':'DM Sans, sans-serif'},
            style_header={'fontWeight':'600','fontSize':'11px','textTransform':'uppercase',
                          'letterSpacing':'0.08em'},
            style_data_conditional=[
                {'if':{'row_index':'odd'},'backgroundColor':'rgba(255,255,255,0.02)'}
            ],
        )

        return html.Div([
            chart_card(dcc.Graph(figure=fig, config={'displayModeBar':False})),
            html.Div([
                html.H4("Drop-off Detail", style={
                    'fontSize':'13px','fontWeight':'600','color':TEXT_SEC,
                    'textTransform':'uppercase','letterSpacing':'0.08em','marginBottom':'12px'
                }),
                table,
            ], style={
                'background':CARD_BG,'border':f'1px solid {CARD_BORDER}',
                'borderRadius':'16px','padding':'20px'
            }),
        ])

    # TAB 6: Ask AI
    elif tab == 'ask_ai':
        return html.Div([
            html.Div([
                html.Div('🤖', style={'fontSize':'36px','marginBottom':'12px'}),
                html.H3("Ask your data anything", style={
                    'fontSize':'20px','fontWeight':'700','color':TEXT_PRI,
                    'margin':'0 0 8px','letterSpacing':'-0.02em'
                }),
                html.P("Type a question in plain English — Claude converts it to SQL and runs it instantly.",
                       style={'fontSize':'13px','color':TEXT_SEC,'margin':'0'}),
            ], style={'textAlign':'center','padding':'32px 0 24px'}),

            html.Div([
                html.Span("Try: ", style={'fontSize':'12px','color':TEXT_SEC,'marginRight':'8px'}),
                *[html.Span(q, style={
                    'fontSize':'12px','color':ACCENT,'background':'rgba(0,229,160,0.08)',
                    'border':'1px solid rgba(0,229,160,0.2)',
                    'borderRadius':'20px','padding':'4px 12px','marginRight':'8px',
                }) for q in [
                    "Revenue by city tier last quarter",
                    "Top 5 products by orders",
                    "Monthly active customers 2024"
                ]],
            ], style={'marginBottom':'20px','display':'flex','alignItems':'center','flexWrap':'wrap','gap':'6px'}),

            html.Div([
                dcc.Input(
                    id='nl-input',
                    type='text',
                    placeholder='e.g. Show me revenue for Tier-2 cities last quarter...',
                    style={
                        'flex':'1','padding':'14px 18px','fontSize':'14px',
                        'borderRadius':'12px','border':f'1px solid {CARD_BORDER}',
                        'background':CARD_BG,'color':TEXT_PRI,
                        'fontFamily':'DM Sans, sans-serif',
                    }
                ),
                html.Button("Run Query →", id='nl-button', style={
                    'marginLeft':'10px','padding':'14px 24px',
                    'background':f'linear-gradient(135deg, {ACCENT}, #00B880)',
                    'color':'#0D0F14','border':'none','borderRadius':'12px',
                    'cursor':'pointer','fontSize':'14px','fontWeight':'700',
                    'fontFamily':'DM Sans, sans-serif',
                    'whiteSpace':'nowrap',
                    'boxShadow':'0 4px 20px rgba(0,229,160,0.3)',
                }),
            ], style={'display':'flex','alignItems':'center','marginBottom':'24px'}),

            html.Div(id='nl-output'),
        ], style={
            'background':CARD_BG,'border':f'1px solid {CARD_BORDER}',
            'borderRadius':'20px','padding':'24px 32px',
        })


# ── NL Query callback ──────────────────────────────────────────────────────────
@app.callback(
    Output('nl-output', 'children'),
    Input('nl-button', 'n_clicks'),
    Input('nl-input', 'value'),
    prevent_initial_call=True,
)
def run_nl_query(n_clicks, question):
    from src.db import query as run_query
    if not question:
        return html.P("Please enter a question.", style={'color':ACCENT3,'fontSize':'13px'})
    try:
        sql = nl_to_sql(question)
        df  = run_query(sql)
        return html.Div([
            html.Div([
                html.Span("Generated SQL", style={
                    'fontSize':'11px','color':ACCENT,'fontWeight':'600',
                    'textTransform':'uppercase','letterSpacing':'0.08em'
                }),
                html.Pre(sql, style={
                    'fontSize':'12px','color':TEXT_SEC,'fontFamily':'DM Mono, monospace',
                    'margin':'8px 0 0','padding':'12px','background':'rgba(0,0,0,0.3)',
                    'borderRadius':'8px','overflowX':'auto','whiteSpace':'pre-wrap'
                }),
            ], style={
                'background':'rgba(108,99,255,0.06)','border':'1px solid rgba(108,99,255,0.15)',
                'borderRadius':'12px','padding':'16px','marginBottom':'16px'
            }),
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': c, 'id': c} for c in df.columns],
                style_table={'overflowX':'auto','borderRadius':'12px','overflow':'hidden'},
                style_cell={'textAlign':'left','padding':'12px 16px','border':'none',
                            'fontFamily':'DM Sans, sans-serif'},
                style_data_conditional=[
                    {'if':{'row_index':'odd'},'backgroundColor':'rgba(255,255,255,0.02)'}
                ],
            )
        ])
    except Exception as e:
        return html.Div([
            html.Span("⚠ Error: ", style={'color':ACCENT3,'fontWeight':'600'}),
            html.Span(str(e), style={'color':TEXT_SEC,'fontSize':'13px'}),
        ], style={
            'background':'rgba(255,107,107,0.08)','border':'1px solid rgba(255,107,107,0.2)',
            'borderRadius':'10px','padding':'12px 16px'
        })


# ── PDF Export callback ────────────────────────────────────────────────────────
@app.callback(
    Output('download-pdf', 'data'),
    Input('download-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def download_pdf(n_clicks):
    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    elements = []

    elements.append(Paragraph("E-commerce Analytics Report", styles['Title']))
    elements.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles['Normal']))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Key Metrics", styles['Heading2']))
    kpi_data = [
        ['Metric', 'Value'],
        ['Total Revenue',   f"Rs.{total_revenue:,.0f}"],
        ['Total Customers', f"{total_customers:,}"],
        ['Avg Order Value', f"Rs.{avg_order_value:,.0f}"],
        ['Churn Rate',      f"{churn_rate:.1f}%"],
    ]
    kpi_table = Table(kpi_data, colWidths=[250, 200])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00E5A0')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.HexColor('#0D0F14')),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 11),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F8F8F8'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 24))

    def add_chart(fig, title):
        elements.append(Paragraph(title, styles['Heading2']))
        img_bytes = pio.to_image(fig, format='png', width=700, height=350, scale=2)
        img = Image(io.BytesIO(img_bytes), width=500, height=250)
        elements.append(img)
        elements.append(Spacer(1, 16))

    fig1 = px.line(revenue_df, x='month', y='revenue', title='Monthly Revenue')
    add_chart(fig1, "Monthly Revenue")

    fig2 = px.bar(revenue_df, x='month', y='mom_growth_pct',
                  title='MoM Growth %', color='mom_growth_pct',
                  color_continuous_scale='RdYlGn')
    add_chart(fig2, "Month-over-Month Growth")

    fig3 = px.bar(cat_df, x='category', y='revenue', title='Revenue by Category')
    add_chart(fig3, "Revenue by Category")

    fig4 = px.bar(rfm_sum_df, x='rfm_segment', y='customers',
                  color='rfm_segment', title='Customers per Segment')
    add_chart(fig4, "RFM Segments")

    doc.build(elements)
    buffer.seek(0)
    return dcc.send_bytes(buffer.read(), filename="ecommerce_report.pdf")


# ── Cache & auto-refresh every hour ───────────────────────────────────────────
CACHE = {
    'revenue': revenue_df,
    'cohorts': cohort_df,
}

def refresh_data():
    global CACHE
    CACHE['revenue'] = get_monthly_revenue()
    CACHE['cohorts'] = get_cohort_pivot()
    print("Data refreshed at", datetime.datetime.now())

scheduler = BackgroundScheduler()
scheduler.add_job(refresh_data, 'interval', hours=1)
scheduler.start()
# Add to dashboard/app.py — Render needs this
server = app.server  # expose Flask server

# requirements.txt must include:
# gunicorn
# dash
# plotly
# pandas
# etc.

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)