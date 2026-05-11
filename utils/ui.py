import streamlit as st
from pathlib import Path


def inject_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f3f7fb;
        }
        .card {
            background-color: #ffffff;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
            margin-bottom: 16px;
        }
        .metric {
            font-size: 32px;
            font-weight: 700;
            color: #0d3b66;
        }
        .event-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .small-text {
            font-size: 13px;
            color: #555;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(alert_count: int, event_count: int, screenshot_count: int):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🚨 Alerts")
        st.markdown(f'<div class="metric">{alert_count}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Events")
        st.markdown(f'<div class="metric">{event_count}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📸 Screenshots")
        st.markdown(f'<div class="metric">{screenshot_count}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_event_cards(summary: dict):
    if not summary:
        return
    cols = st.columns(2)
    items = list(summary.items())
    for index, (label, data) in enumerate(items):
        with cols[index % 2]:
            st.markdown(
                f"""
                <div class="event-card">
                    <strong>{label.upper()}</strong><br>
                    Count: {data['count']}<br>
                    Max Threat: {data['max_threat']}%<br>
                    Risk: {'HIGH' if data['risk'] == 'HIGH' else 'LOW'}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_screenshot_grid(image_paths):
    if not image_paths:
        return
    st.markdown("### 📸 Alert Screenshots")
    cols = st.columns(3)
    for idx, path in enumerate(image_paths):
        try:
            image = Path(path)
            if image.exists():
                st.image(str(image), caption=image.name, use_column_width=True)
            else:
                st.markdown(f"*Missing image: {path}*")
        except Exception:
            st.markdown(f"*Unable to load screenshot: {path}*")
