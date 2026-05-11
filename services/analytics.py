import plotly.graph_objects as go


def build_threat_timeline(threat_history):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(threat_history))),
            y=threat_history,
            mode="lines+markers",
            marker=dict(color="#1f77b4"),
            line=dict(shape="spline", smoothing=0.7),
            name="Threat Score",
        )
    )
    fig.update_layout(
        title="Threat Timeline",
        xaxis_title="Frame",
        yaxis_title="Threat %",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def build_risk_pie(summary):
    labels = []
    values = []
    for label, data in summary.items():
        labels.append(label)
        values.append(data["count"])
    if not labels:
        return None
    fig = go.Figure(
        go.Pie(labels=labels, values=values, hole=0.4, textinfo="percent+label")
    )
    fig.update_layout(title="Event Distribution", margin=dict(t=40, b=20, l=20, r=20))
    return fig
