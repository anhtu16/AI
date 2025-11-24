import io
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from model_service import ABSAService, AspectPrediction, SentimentPrediction

pio.templates.default = "plotly_dark"


@st.cache_resource(show_spinner=True)
def load_service() -> ABSAService:
    return ABSAService()


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
            .stApp {
                font-family: "Space Grotesk", sans-serif;
                background: #03060f;
                color: #f5f6fb;
            }
            .stApp::before,
            .stApp::after {
                content: "";
                position: fixed;
                width: 120vw;
                height: 120vh;
                top: -10vh;
                left: -10vw;
                background: conic-gradient(
                    from 180deg,
                    rgba(125, 211, 252, 0.24),
                    rgba(192, 132, 252, 0.18),
                    rgba(244, 114, 182, 0.18),
                    rgba(125, 211, 252, 0.24)
                );
                filter: blur(120px);
                animation: aurora 15s linear infinite;
                z-index: 0;
            }
            .stApp::after {
                animation-direction: reverse;
                opacity: 0.85;
            }
            @keyframes aurora {
                0% { transform: translate(-6%, -4%) scale(0.95) rotate(0deg); }
                35% { transform: translate(8%, 6%) scale(1.05) rotate(120deg); }
                70% { transform: translate(-4%, 10%) scale(1.1) rotate(240deg); }
                100% { transform: translate(-6%, -4%) scale(0.95) rotate(360deg); }
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                position: relative;
                z-index: 1;
            }
            body::before {
                content: "";
                position: fixed;
                inset: 0;
                background: radial-gradient(circle at 10% 20%, rgba(125,211,252,0.25), transparent 45%),
                            radial-gradient(circle at 80% 10%, rgba(248,113,113,0.2), transparent 35%),
                            radial-gradient(circle at 50% 80%, rgba(192,132,252,0.2), transparent 40%);
                filter: blur(40px);
                z-index: 0;
            }
            .main .block-container {
                position: relative;
                z-index: 1;
            }
            .ai-glow {
                position: relative;
            }
            .ai-glow::after {
                content: "";
                position: absolute;
                top: -40px;
                right: -60px;
                width: 140px;
                height: 140px;
                background: radial-gradient(circle, rgba(94,234,212,0.35), transparent 65%);
                filter: blur(8px);
                animation: pulse 4s ease-in-out infinite;
            }
            @keyframes pulse {
                0% { transform: scale(0.9); opacity: 0.6; }
                50% { transform: scale(1.15); opacity: 1; }
                100% { transform: scale(0.9); opacity: 0.6; }
            }
            .glass-card {
                background: rgba(11, 15, 28, 0.65);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 24px;
                padding: 1.8rem;
                box-shadow: 0 40px 80px rgba(5, 6, 24, 0.65);
                backdrop-filter: blur(18px);
            }
            .hero-title {
                font-size: 2.8rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
                background: linear-gradient(90deg, #7dd3fc, #c084fc, #f472b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .hero-sub {
                color: #b5bfd8;
                font-size: 1.1rem;
                margin-bottom: 0.6rem;
            }
            .badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.4rem 0.9rem;
                border-radius: 999px;
                font-size: 0.85rem;
                background: rgba(125, 211, 252, 0.2);
                color: #7dd3fc;
                border: 1px solid rgba(125, 211, 252, 0.3);
            }
            .pill {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.35rem 0.8rem;
                border-radius: 999px;
                font-size: 0.85rem;
            }
            .pill-POS {background: rgba(16,185,129,.2); color:#34d399;}
            .pill-NEG {background: rgba(248,113,113,.2); color:#f87171;}
            .pill-NEU {background: rgba(148,163,184,.2); color:#cbd5f5;}
            div[data-testid="stMetric"] {
                background: rgba(255,255,255,0.06);
                border-radius: 22px;
                padding: 1.2rem;
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
            }
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                border-radius: 999px;
                padding: 0.4rem 1.2rem;
                border: 1px solid rgba(255,255,255,0.08);
                margin-right: 0.6rem;
            }
            .stTabs [aria-selected="true"] {
                background: rgba(255,255,255,0.1);
            }
            .stTabs {
                margin-top: 1.4rem;
                margin-bottom: 2rem;
            }
            .stTabs [role="tablist"] {
                padding-bottom: 0.6rem;
                border-bottom: 1px solid rgba(255,255,255,0.08);
            }
            table {
                border-radius: 18px;
                overflow: hidden;
            }
            .ai-overlay {
                position: relative;
                isolation: isolate;
            }
            .ai-overlay::before {
                content: "";
                position: absolute;
                inset: -20px;
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.03);
                border-radius: 32px;
                backdrop-filter: blur(20px);
                z-index: -1;
                animation: floaty 12s ease-in-out infinite;
            }
            @keyframes floaty {
                0% { transform: translateY(0px); }
                50% { transform: translateY(12px); }
                100% { transform: translateY(0px); }
            }
            .dashboard-animate {
                animation: dashboardFade 0.9s cubic-bezier(0.16, 1, 0.3, 1);
            }
            @keyframes dashboardFade {
                0% { opacity: 0; transform: translateY(30px) scale(0.96); }
                100% { opacity: 1; transform: translateY(0) scale(1); }
            }
            .action-card {
                padding: 1.2rem;
                border-radius: 20px;
                border: 1px solid rgba(255,255,255,0.06);
                background: rgba(13,16,29,0.75);
                backdrop-filter: blur(16px);
                box-shadow: 0 20px 40px rgba(2,6,23,0.35);
                margin-bottom: 1.2rem;
            }
            .severity-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.3rem;
                padding: 0.2rem 0.7rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .severity-critical { background: rgba(239,68,68,0.25); color: #f87171; }
            .severity-high { background: rgba(251,146,60,0.25); color: #fb923c; }
            .severity-medium { background: rgba(234,179,8,0.2); color: #fde047; }
            .severity-low { background: rgba(34,197,94,0.2); color: #6ee7b7; }
            .action-card h4 { margin: 0 0 0.4rem 0; }
            .action-card p { margin: 0; color: #cbd5f5; font-size: 0.95rem; }
            .action-card small { color: #94a3b8; }
            .playbook-list li { margin-bottom: 0.3rem; }
            .playbook-list strong { color: #fefce8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def manual_analysis(service: ABSAService) -> None:
    with st.container():
        st.subheader("⚡ Phân tích nhanh")
        col_left, col_right = st.columns([3, 1.2], gap="large")
        with col_left:
            text = st.text_area(
                "Nhập câu cần phân tích",
                height=180,
                label_visibility="collapsed",
                placeholder="Nhập nhận xét của khách hàng...",
            )
        with col_right:
            st.caption("Ngưỡng xác định aspect (sigmoid score)")
            threshold = st.slider(
                "",
                min_value=0.1,
                max_value=0.9,
                value=service.aspect_threshold,
                step=0.05,
                label_visibility="collapsed",
            )
        service.update_threshold(threshold)

        if st.button("Phân tích câu", use_container_width=True, type="primary"):
            if not text.strip():
                st.warning("Vui lòng nhập nội dung.")
                return

            with st.spinner("Đang phân tích..."):
                result = service.analyze_text(text)

            aspects = result["aspects"]
            sentiment: SentimentPrediction = result["sentiment"]

            if sentiment:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Sentiment tổng thể", sentiment.label, f"{sentiment.score:.2f}")
                with col_b:
                    st.metric("Số aspect phát hiện", len(aspects))
                with col_c:
                    st.metric("Ngưỡng hiện tại", f"{service.aspect_threshold:.2f}")
                st.markdown(
                    f"<div class='pill pill-{sentiment.label.upper()}'>Confidence {sentiment.score:.2f}</div>",
                    unsafe_allow_html=True,
                )

            st.write("**Insight theo từng aspect**")
            if aspects:
                table_data = []
                for a in aspects:
                    table_data.append(
                        {
                            "Aspect": a.label,
                            "Aspect score": f"{a.score:.2f}",
                            "Sentiment": a.sentiment.label if a.sentiment else "-",
                            "Sentiment score": (
                                f"{a.sentiment.score:.2f}"
                                if a.sentiment
                                else "-"
                            ),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(table_data),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Không tìm thấy aspect nào với ngưỡng hiện tại.")


def _read_uploaded_file(uploaded) -> pd.DataFrame:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in [".xls", ".xlsx"]:
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


def batch_analysis(service: ABSAService) -> None:
    st.subheader("📁 Phân tích file")
    uploaded = st.file_uploader("Upload file CSV hoặc Excel", type=["csv", "xls", "xlsx"])
    text_column = st.text_input("Tên cột chứa câu cần phân tích", value="text")

    if uploaded and st.button("Phân tích file", use_container_width=True):
        try:
            df = _read_uploaded_file(uploaded)
        except Exception as exc:
            st.error(f"Không đọc được file: {exc}")
            return

        if text_column not in df.columns:
            st.error(f"Không tìm thấy cột '{text_column}' trong file.")
            return

        records = []
        with st.spinner("Đang chạy mô hình trên toàn bộ dữ liệu..."):
            for text in df[text_column].fillna("").tolist():
                result = service.analyze_text(str(text))
                aspects: List[AspectPrediction] = result["aspects"]
                sentiment: SentimentPrediction = result["sentiment"]
                aspect_details = []
                for aspect in aspects:
                    aspect_details.append(
                        {
                            "aspect": aspect.label,
                            "aspect_score": aspect.score,
                            "sentiment": aspect.sentiment.label
                            if aspect.sentiment
                            else None,
                            "sentiment_score": aspect.sentiment.score
                            if aspect.sentiment
                            else None,
                        }
                    )

                if aspect_details:
                    aspects_display = "; ".join(
                        f"{d['aspect']} ({d.get('sentiment', '-')}, "
                        f"{(d.get('sentiment_score') or 0):.2f})"
                        for d in aspect_details
                    )
                else:
                    aspects_display = "-"

                records.append(
                    {
                        text_column: text,
                        "sentiment_label": sentiment.label if sentiment else None,
                        "sentiment_score": sentiment.score if sentiment else None,
                        "aspects_display": aspects_display,
                        "aspects_detail": aspect_details,
                    }
                )

        analysis_df = pd.DataFrame(records)
        st.session_state["analysis_df"] = analysis_df

        st.success("Phân tích hoàn tất!")
        display_df = analysis_df.drop(columns=["aspects_detail"])
        display_df = display_df.rename(columns={"aspects_display": "aspects"})
        st.dataframe(display_df, use_container_width=True)

        csv_buffer = io.StringIO()
        analysis_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Tải kết quả CSV",
            data=csv_buffer.getvalue(),
            file_name="analysis_results.csv",
            mime="text/csv",
        )


def dashboard() -> None:
    st.subheader("📊 Dashboard kết quả")
    analysis_df: pd.DataFrame | None = st.session_state.get("analysis_df")
    if analysis_df is None or analysis_df.empty:
        st.info("Hãy phân tích file để có dữ liệu hiển thị.")
        return

    st.markdown("<div class='dashboard-animate'>", unsafe_allow_html=True)

    total_rows = len(analysis_df)
    pos = (analysis_df["sentiment_label"] == "POS").sum()
    neg = (analysis_df["sentiment_label"] == "NEG").sum()
    neu = (analysis_df["sentiment_label"] == "NEU").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng review", total_rows)
    c2.metric("Positive", pos)
    c3.metric("Neutral", neu)
    c4.metric("Negative", neg)

    col1, col2 = st.columns((1, 1))

    sentiment_counts = (
        analysis_df["sentiment_label"].value_counts().reset_index()
    )
    sentiment_counts.columns = ["sentiment", "count"]
    fig_sentiment = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map={
            "POS": "#059669",
            "NEU": "#475569",
            "NEG": "#dc2626",
        },
        hole=0.35,
    )
    gradient_colors = {
        "POS": ["rgba(5,150,105,0.95)", "rgba(16,185,129,0.7)"],
        "NEU": ["rgba(71,85,105,0.95)", "rgba(100,116,139,0.8)"],
        "NEG": ["rgba(220,38,38,0.95)", "rgba(248,113,113,0.8)"],
    }
    ordered_colors = []
    for label in sentiment_counts["sentiment"]:
        shades = gradient_colors.get(label, ["#94a3b8"])
        ordered_colors.append(shades[0])
    pulls = [0.04 if label == "NEG" else 0 for label in sentiment_counts["sentiment"]]
    if len(sentiment_counts) < 3:
        pulls = [0 for _ in sentiment_counts["sentiment"]]
    fig_sentiment.update_traces(
        marker=dict(line=dict(color="#020617", width=2), colors=ordered_colors),
        rotation=25,
        pull=pulls,
    )
    fig_sentiment.update_traces(textposition="inside", textinfo="percent+label")
    fig_sentiment.update_layout(transition_duration=700)
    col1.plotly_chart(fig_sentiment, use_container_width=True)

    aspect_series = analysis_df["aspects_detail"].explode()
    if aspect_series.notna().any():
        aspect_counts = (
            aspect_series.dropna()
            .apply(lambda x: x.get("aspect"))
            .value_counts()
            .reset_index()
        )
        aspect_counts.columns = ["aspect", "count"]
        fig_aspects = px.bar(
            aspect_counts,
            x="count",
            y="aspect",
            orientation="h",
            color="count",
            color_continuous_scale="Agsunset",
        )
        fig_aspects.update_layout(coloraxis_showscale=False)
        fig_aspects.update_layout(transition_duration=700)
        col2.plotly_chart(fig_aspects, use_container_width=True)
    else:
        col2.info("Không có aspect nào vượt ngưỡng đã đặt.")

    # Additional dashboard elements
    col3, col4 = st.columns((1, 1))

    timeline = analysis_df.copy()
    timeline = timeline.dropna(subset=["sentiment_score"])
    timeline["index"] = range(1, len(timeline) + 1)
    line_chart = px.line(
        timeline,
        x="index",
        y="sentiment_score",
        color="sentiment_label",
        color_discrete_map={
            "POS": "#34d399",
            "NEU": "#cbd5f5",
            "NEG": "#f87171",
        },
        markers=True,
    )
    line_chart.update_traces(line=dict(width=3))
    line_chart.update_layout(
        transition_duration=700, xaxis_title="Review #", yaxis_title="Confidence"
    )
    col3.plotly_chart(line_chart, use_container_width=True)

    aspect_detail_df = (
        analysis_df.explode("aspects_detail").dropna(subset=["aspects_detail"])
    )
    if not aspect_detail_df.empty:
        detail = pd.json_normalize(aspect_detail_df["aspects_detail"])
        detail["sentiment"].fillna("NEU", inplace=True)
        aspect_sentiment_counts = (
            detail.groupby(["aspect", "sentiment"])
            .size()
            .reset_index(name="count")
        )
        stacked_chart = px.bar(
            aspect_sentiment_counts,
            x="aspect",
            y="count",
            color="sentiment",
            color_discrete_map={
                "POS": "#34d399",
                "NEU": "#cbd5f5",
                "NEG": "#f87171",
            },
            barmode="stack",
        )
        stacked_chart.update_layout(
            transition_duration=700,
            legend_title="Sentiment",
            xaxis_title="Aspect",
            yaxis_title="Số lần xuất hiện",
        )
        col4.plotly_chart(stacked_chart, use_container_width=True)
    else:
        col4.info("Chưa có dữ liệu aspect để vẽ biểu đồ phân bố.")

    st.markdown("</div>", unsafe_allow_html=True)


def action_center() -> None:
    st.subheader("🎯 Action Center · Ưu tiên hành động")
    analysis_df: pd.DataFrame | None = st.session_state.get("analysis_df")
    if analysis_df is None or analysis_df.empty:
        st.info("Hãy phân tích file trước khi tạo gợi ý hành động.")
        return

    exploded = analysis_df.explode("aspects_detail").dropna(subset=["aspects_detail"])
    if exploded.empty:
        st.info("Chưa có dữ liệu aspect để tổng hợp khuyến nghị.")
        return

    detail = pd.json_normalize(exploded["aspects_detail"])
    detail.columns = [col.split(".")[-1] for col in detail.columns]
    detail["sentiment"] = detail["sentiment"].fillna("NEU")
    detail["sentiment_upper"] = detail["sentiment"].str.upper()
    detail["is_neg"] = detail["sentiment_upper"].isin(["NEG", "NEGATIVE"])
    detail["is_pos"] = detail["sentiment_upper"].isin(["POS", "POSITIVE"])
    detail["score"] = detail["sentiment_score"].fillna(0)

    stats = (
        detail.groupby("aspect")
        .agg(
            mentions=("aspect", "count"),
            neg=("is_neg", "sum"),
            pos=("is_pos", "sum"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )
    stats["neg_ratio"] = stats["neg"] / stats["mentions"]
    stats["pos_ratio"] = stats["pos"] / stats["mentions"]
    if stats["mentions"].max() > 0:
        stats["priority_score"] = (
            0.7 * stats["neg_ratio"] + 0.3 * (stats["mentions"] / stats["mentions"].max())
        )
    else:
        stats["priority_score"] = stats["neg_ratio"]

    st.markdown("##### ⚙️ Tham số ưu tiên")
    col_ctrl1, col_ctrl2 = st.columns(2)
    max_mentions_value = max(1, int(stats["mentions"].max()))
    total_aspects = max(1, stats["aspect"].nunique())
    top_slider_min = 1 if total_aspects < 3 else 3
    top_slider_max = max(top_slider_min, min(8, total_aspects))
    with col_ctrl1:
        min_mentions = st.slider(
            "Số lượt nhắc tối thiểu của một aspect",
            min_value=1,
            max_value=max_mentions_value,
            value=max(1, max_mentions_value // 5),
        )
    with col_ctrl2:
        top_n = st.slider(
            "Số lượng khía cạnh hiển thị",
            min_value=top_slider_min,
            max_value=top_slider_max,
            value=min(5, top_slider_max),
        )

    filtered_stats = stats[stats["mentions"] >= min_mentions].copy()
    if filtered_stats.empty:
        st.info("Không có aspect nào đạt số lượt nhắc tối thiểu. Giảm ngưỡng để xem thêm dữ liệu.")
        return

    def severity_label(ratio: float) -> str:
        if ratio >= 0.6:
            return "critical"
        if ratio >= 0.4:
            return "high"
        if ratio >= 0.25:
            return "medium"
        return "low"

    urgent = (
        filtered_stats.sort_values(["priority_score"], ascending=False)
        .head(top_n)
        .copy()
    )
    urgent["severity"] = urgent["neg_ratio"].apply(severity_label)

    st.markdown("##### 🔥 Khía cạnh cần ưu tiên xử lý")
    if urgent.empty:
        st.info("Không có khía cạnh nào thoả điều kiện lọc hiện tại.")
    else:
        urgent_cols = st.columns(len(urgent))
        for col, (_, row) in zip(urgent_cols, urgent.iterrows()):
            with col:
                st.markdown(
                    f"""
                    <div class="action-card">
                        <div class="severity-chip severity-{row['severity']}">
                            {row['severity'].upper()}
                        </div>
                        <h4>{row['aspect']}</h4>
                        <p>{int(row['neg'])} phản hồi NEG ({row['neg_ratio']:.0%})</p>
                        <small>Confidence trung bình {row['avg_score']:.2f} · {int(row['mentions'])} lượt nhắc</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    opportunity = (
        filtered_stats.sort_values(["pos_ratio", "mentions"], ascending=False)
        .head(top_n)
        .copy()
    )

    st.markdown("##### 🌱 Cơ hội nổi bật")
    if opportunity.empty:
        st.info("Chưa có cơ hội nổi bật với bộ lọc hiện tại.")
    else:
        opp_cols = st.columns(len(opportunity))
        for col, (_, row) in zip(opp_cols, opportunity.iterrows()):
            sentiment_badge = "severity-low" if row["pos_ratio"] >= 0.5 else "severity-medium"
            with col:
                st.markdown(
                    f"""
                    <div class="action-card">
                        <div class="severity-chip {sentiment_badge}">
                            POSITIVE
                        </div>
                        <h4>{row['aspect']}</h4>
                        <p>{int(row['pos'])} phản hồi POS ({row['pos_ratio']:.0%})</p>
                        <small>Confidence trung bình {row['avg_score']:.2f} · {int(row['mentions'])} lượt nhắc</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    text_columns = [
        col
        for col in analysis_df.columns
        if col not in {"sentiment_label", "sentiment_score", "aspects_display", "aspects_detail"}
    ]
    sample_col = text_columns[0] if text_columns else None

    st.markdown("##### 🧩 Gợi ý hành động cụ thể")
    suggestions = []
    owner_map = {
        "PRICE": "Sales & Pricing",
        "GENERAL": "CSKH",
        "BATTERY": "Hardware",
        "CAMERA": "R&D Camera",
        "PERFORMANCE": "Performance Eng",
        "DESIGN": "Product Design",
        "FEATURES": "Product Team",
        "SCREEN": "Display Team",
        "STORAGE": "Hardware",
        "SER&ACC": "Service & Accessories",
    }

    def build_action_row(row, focus: str) -> dict:
        owner = owner_map.get(row["aspect"], "Product Owner")
        severity = row.get("severity", "medium").capitalize()
        if focus == "neg":
            next_step = "Điều tra nguyên nhân, phản hồi khách hàng và đưa phương án cải tiến."
        else:
            next_step = "Tận dụng insight tích cực cho chiến dịch marketing/upsell."
        return {
            "Aspect": row["aspect"],
            "Owner đề xuất": owner,
            "Severity": severity,
            "Mentions": int(row["mentions"]),
            "Next step": next_step,
        }

    action_plan_rows = [build_action_row(row, "neg") for _, row in urgent.iterrows()]
    action_plan_rows += [
        build_action_row(row, "pos")
        for _, row in opportunity.iterrows()
        if row["pos"] > 0
    ]

    st.markdown("##### 🧩 Playbook hành động")
    if sample_col and not exploded.empty:
        negative_examples = exploded[
            exploded["aspects_detail"].apply(
                lambda d: d.get("sentiment") in ["NEG", "NEGATIVE"]
            )
        ]
        if not negative_examples.empty:
            sample_text = negative_examples.iloc[0][sample_col]
            st.markdown(
                f"> **Example complaint:** “{sample_text}”",
            )

    if action_plan_rows:
        st.table(pd.DataFrame(action_plan_rows))
    else:
        st.info("Chưa đủ dữ liệu để tạo playbook hành động.")

    summary_csv = filtered_stats.to_csv(index=False)
    st.download_button(
        "Tải báo cáo ưu tiên (CSV)",
        data=summary_csv,
        file_name="action_center_summary.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="Aspect-based Sentiment Analysis",
        layout="wide",
        page_icon=":bar_chart:",
    )
    inject_custom_css()
    hero = st.container()
    with hero:
        st.markdown(
            """
            <div class="glass-card ai-glow">
                <div class="hero-title">Nhóm 10 · Trí tuệ nhân tạo trong kinh doanh</div>
                <div class="hero-sub">
                    Hệ thống phân tích cảm xúc đa khía cạnh giúp doanh nghiệp hiểu sâu insight khách hàng.
                </div>
                <div class="badge">Aspect-based Sentiment Dashboard · ViSFD Dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    service = load_service()

    tab_manual, tab_file, tab_dashboard, tab_actions = st.tabs(
        ["🔍 Phân tích câu", "📁 Phân tích file", "📊 Dashboard", "🎯 Action Center"]
    )

    with tab_manual:
        with st.container():
            manual_analysis(service)
    with tab_file:
        with st.container():
            batch_analysis(service)
    with tab_dashboard:
        with st.container():
            dashboard()
    with tab_actions:
        with st.container():
            action_center()


if __name__ == "__main__":
    main()

