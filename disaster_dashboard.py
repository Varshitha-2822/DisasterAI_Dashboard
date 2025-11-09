import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------
# Load correlation matrix
# ----------------------------------------------------------------
import os
FILE_PATH = os.path.join(os.path.dirname(__file__), "correlation_matrix.csv")

corr_df = pd.read_csv(FILE_PATH, index_col=0)

# ----------------------------------------------------------------
# Initialize Dash App
# ----------------------------------------------------------------
app = Dash(__name__, title="DisasterAI Dashboard")

app.layout = html.Div(style={"backgroundColor": "#F4F6F7", "padding": "20px"}, children=[
    html.H1("🌍 DisasterAI Interactive Dashboard", style={"textAlign": "center", "color": "#154360"}),
    html.Hr(),

    dcc.Tabs([
        dcc.Tab(label="📊 Correlation Heatmap", children=[
            dcc.Graph(figure=px.imshow(
                corr_df, text_auto=True, color_continuous_scale="YlGnBu",
                title="Feature Correlation Matrix"
            ).update_layout(title_x=0.5, font=dict(size=12)))
        ]),
        dcc.Tab(label="📈 Pairwise Correlation", children=[
            html.H3("Select Feature:", style={"textAlign": "center"}),
            dcc.Dropdown(
                id="metric_dropdown",
                options=[{"label": col, "value": col} for col in corr_df.columns],
                value=corr_df.columns[0],
                clearable=False,
                style={"width": "50%", "margin": "auto"},
            ),
            dcc.Graph(id="metric_bar_chart"),
            html.Div(id="insight_box", style={
                "textAlign": "center", "backgroundColor": "#EBF5FB",
                "padding": "15px", "borderRadius": "10px",
                "width": "70%", "margin": "20px auto", "fontSize": "18px"
            })
        ])
    ])
])

# ----------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------
@app.callback(
    [Output("metric_bar_chart", "figure"), Output("insight_box", "children")],
    Input("metric_dropdown", "value")
)
def update_metric_chart(selected_metric):
    series = corr_df[selected_metric].sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=series.values, y=series.index, orientation="h", marker_color="#1ABC9C"))
    fig.update_layout(title=f"Correlation with '{selected_metric}'", xaxis_title="r", height=500)

    top_corr = series.drop(selected_metric).nlargest(1)
    low_corr = series.drop(selected_metric).nsmallest(1)
    insight = f"'{selected_metric}' correlates most with '{top_corr.index[0]}' (r={top_corr.values[0]:.2f}) and least with '{low_corr.index[0]}' (r={low_corr.values[0]:.2f})."
    return fig, insight

if __name__ == "__main__":
        app.run(debug=True)
