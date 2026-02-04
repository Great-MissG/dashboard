import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import pgeocode
import re

# Cached Nominatim instance for US (initialized on first use)
NOMI_US = None

st.set_page_config(page_title="FIMILE Dashboard", layout="wide")

# 本文件为只读 Dashboard 首页，数据源来自 Google Sheet
# UI：顶部固定区（模块1 KPI + 全局时间选择器）+ Tabs区域（订单与营收 + 时效）

TARGET_REVENUE = 49_500_000  # USD


def _fmt_currency(value: float) -> str:
    """格式化货币显示"""
    try:
        return f"${value:,.0f}"
    except Exception:
        return "$0"


def _fmt_duration_hours(td) -> str:
    """格式化时间差为小时或天"""
    if pd.isna(td) or td is None:
        return "-"
    try:
        total_seconds = td.total_seconds()
        hours = total_seconds / 3600
        days = hours / 24
        if days >= 1:
            return f"{days:.1f} 天"
        else:
            return f"{hours:.1f} 小时"
    except Exception:
        return "-"


def _find_state_col(df: pd.DataFrame, direction: str):
    """
    Try to find a state column by heuristic matching.
    direction: "pickup" or "delivery"
    """
    if df is None or len(df.columns) == 0:
        return None
    direction = direction.lower().strip()
    if direction not in ["pickup", "delivery"]:
        return None
    dir_keywords = ["pickup", "origin", "shipper", "from"] if direction == "pickup" else ["delivery", "destination", "consignee", "to", "drop"]
    for col in df.columns:
        norm = re.sub(r"[^a-z0-9]", "", str(col).lower())
        if "state" in norm and any(k in norm for k in dir_keywords):
            return col
    return None


def _use_tracking_id_for_count(df: pd.DataFrame) -> bool:
    """
    Decide whether tracking_id is reliable for order counting.
    If tracking_id is missing/blank or collapsed to a single value while there are
    multiple rows, fall back to row count.
    """
    if df is None or len(df) == 0:
        return False
    if "tracking_id" not in df.columns:
        return False
    tracking_ids = df["tracking_id"].astype(str).str.strip()
    tracking_ids = tracking_ids[tracking_ids != ""]
    unique_count = tracking_ids.nunique()
    if len(df) <= 1:
        return unique_count == 1
    return unique_count >= 2


def clean_zip5(x):
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().replace("'", "")
    if s == "":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return None
    zip5 = digits[:5]
    return zip5.zfill(5)


def load_sheet_to_df():
    """
    Load Google Sheet into a pandas DataFrame using credentials from `st.secrets`.
    Returns (df, error_message). On success error_message is None.
    """
    try:
        # Support two shapes: gcp_service_account as JSON string or dict
        sa = st.secrets.get("gcp_service_account")
        if not sa:
            return None, "gcp_service_account not found in st.secrets"

        if isinstance(sa, str):
            creds_dict = json.loads(sa)
        else:
            creds_dict = dict(sa)

        sheet_id = st.secrets.get("sheet_id") or st.secrets.get("gcp_service_account", {}).get("sheet_id")
        worksheet_name = st.secrets.get("worksheet_name")

        if not sheet_id:
            return None, "sheet_id missing in st.secrets"

        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)

        if worksheet_name:
            ws = sh.worksheet(worksheet_name)
        else:
            ws = sh.sheet1

        records = ws.get_all_records()
        df = pd.DataFrame(records)
        # Clean column names
        df.columns = df.columns.str.strip()
        return df, None
    except Exception as e:
        return None, str(e)


def enrich_df_with_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理函数：为 DataFrame 添加 pickup_state、delivery_state、state_pair 字段。
    使用 session state 缓存结果，避免重复计算 pgeocode 查询。
    """
    if df is None or len(df) == 0:
        return df
    
    # 使用 session state 缓存，key 基于 DataFrame 的 id
    df_id = id(df)
    cache_key = f"df_enriched_{df_id}"
    
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # 复制 DataFrame 以避免修改原始数据
    df_enriched = df.copy()
    
    # 初始化 state 列（尝试从现有列推断）
    if "pickup_state" not in df_enriched.columns:
        pickup_col = _find_state_col(df_enriched, "pickup")
        if pickup_col:
            df_enriched["pickup_state"] = df_enriched[pickup_col]
        else:
            df_enriched["pickup_state"] = None
    if "delivery_state" not in df_enriched.columns:
        delivery_col = _find_state_col(df_enriched, "delivery")
        if delivery_col:
            df_enriched["delivery_state"] = df_enriched[delivery_col]
        else:
            df_enriched["delivery_state"] = None
    
    # 清洗 ZIP5
    if "pickup_address_zipcode" in df_enriched.columns:
        df_enriched["pickup_zip5"] = df_enriched["pickup_address_zipcode"].apply(clean_zip5)
    else:
        df_enriched["pickup_zip5"] = None
    
    if "delivery_address_zipcode" in df_enriched.columns:
        df_enriched["delivery_zip5"] = df_enriched["delivery_address_zipcode"].apply(clean_zip5)
    else:
        df_enriched["delivery_zip5"] = None
    
    # 使用 pgeocode 映射 ZIP → State
    global NOMI_US
    if NOMI_US is None:
        NOMI_US = pgeocode.Nominatim("US")
    
    try:
        # Pickup
        pickup_zip5_clean = df_enriched["pickup_zip5"].dropna().astype(str).str.zfill(5).unique().tolist()
        if len(pickup_zip5_clean) > 0:
            pickup_query_df = NOMI_US.query_postal_code(pickup_zip5_clean)
            pickup_zip_to_state = dict(zip(pickup_query_df["postal_code"].astype(str).str.zfill(5), pickup_query_df["state_code"].astype(str)))
            pickup_blank = df_enriched["pickup_state"].isna() | df_enriched["pickup_state"].astype(str).str.strip().eq("")
            df_enriched.loc[pickup_blank, "pickup_state"] = df_enriched.loc[pickup_blank, "pickup_zip5"].apply(
                lambda z: pickup_zip_to_state.get(str(z).zfill(5)) if pd.notna(z) else None
            )
        
        # Delivery
        delivery_zip5_clean = df_enriched["delivery_zip5"].dropna().astype(str).str.zfill(5).unique().tolist()
        if len(delivery_zip5_clean) > 0:
            delivery_query_df = NOMI_US.query_postal_code(delivery_zip5_clean)
            delivery_zip_to_state = dict(zip(delivery_query_df["postal_code"].astype(str).str.zfill(5), delivery_query_df["state_code"].astype(str)))
            delivery_blank = df_enriched["delivery_state"].isna() | df_enriched["delivery_state"].astype(str).str.strip().eq("")
            df_enriched.loc[delivery_blank, "delivery_state"] = df_enriched.loc[delivery_blank, "delivery_zip5"].apply(
                lambda z: delivery_zip_to_state.get(str(z).zfill(5)) if pd.notna(z) else None
            )
    except Exception as e:
        # 映射失败，保留 None
        pass
    
    # 生成 state_pair
    df_enriched["state_pair"] = df_enriched["pickup_state"].astype(str) + "-" + df_enriched["delivery_state"].astype(str)
    
    # 缓存结果
    st.session_state[cache_key] = df_enriched
    
    return df_enriched


def render_module1_kpis(df: pd.DataFrame):
    """
    顶部固定区 - 模块1：4个KPI指标
    - 使用全量df（不受时间选择器影响）
    - 年度目标营收 / 累计完成营收 / 完成进度 / 数据更新至
    """
    # Ensure columns exist
    for col in ["order_time", "delivery_time", "Total shipping fee"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Completed revenue: only count rows with delivery_time non-empty
    delivered_mask = df["delivery_time"].notna() & (df["delivery_time"].astype(str).str.strip() != "")
    fees = df.loc[delivered_mask, "Total shipping fee"].astype(str).str.replace("[$,]", "", regex=True).str.strip()
    completed = float(pd.to_numeric(fees, errors="coerce").fillna(0.0).sum())

    target_display = _fmt_currency(TARGET_REVENUE)
    completed_display = _fmt_currency(completed)
    progress_display = f"{(completed / TARGET_REVENUE * 100):.1f}%" if TARGET_REVENUE else "0%"

    # Data updated timestamp: latest order_time if present
    order_times = pd.to_datetime(df.get("order_time"), utc=True, errors="coerce")
    if order_times.dropna().shape[0] > 0:
        latest = order_times.dropna().max()
        data_updated_str = latest.strftime("%Y-%m-%d %H:%M UTC")
    else:
        data_updated_str = "-"

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        st.metric(label="年度目标营收", value=target_display)

    with c2:
        st.metric(label="累计完成营收", value=completed_display)
        st.caption("仅统计 delivery_time 非空")

    with c3:
        st.metric(label="完成进度", value=progress_display)

    with c4:
        st.write("**数据更新至**")
        st.write(f"<div style='font-size: 20px; font-weight: bold;'>{data_updated_str}</div>", unsafe_allow_html=True)


def render_global_date_filter(df: pd.DataFrame):
    """
    全局日期选择器（顶部固定区第二部分）
    返回 (start_date, end_date, df_range)
    df_range 用于所有 Tabs 的数据
    """
    # Parse order_time to determine date range
    order_times = pd.to_datetime(df.get("order_time"), utc=True, errors="coerce")
    order_dates = order_times.dt.date
    
    valid_dates = order_dates.dropna()
    if len(valid_dates) > 0:
        min_date = valid_dates.min()
        max_date = valid_dates.max()
    else:
        min_date = datetime.now().date()
        max_date = datetime.now().date()

    # Default: start of current month, end of max_date in data
    today = datetime.now().date()
    first_day_of_month = today.replace(day=1)
    
    default_start = max(min_date, min(first_day_of_month, max_date))
    default_end = max(min_date, min(max_date, max_date))
    
    if default_start > default_end:
        default_start = default_end

    # Time range selector
    st.subheader("⏰ 统计时间范围")
    date_range = st.date_input(
        "选择时间范围",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date,
        key="global_date_filter"
    )

    # Handle date_input output (can be single date or tuple)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range

    # Filter df by time range (inclusive on both ends)
    # Convert to UTC timestamps for filtering
    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
    end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize("UTC")
    
    mask_time_range = (order_times >= start_ts) & (order_times < end_ts)
    df_range = df.loc[mask_time_range].copy()
    
    return start_date, end_date, df_range


def render_tab_orders_revenue(df_range: pd.DataFrame):
    """
    Tab1：订单与营收
    包含：KPI指标 + 周订单量柱状图 + 周营收柱状图 + 客户饼图
    """
    if len(df_range) == 0:
        st.info("所选时间范围内暂无数据")
        return

    # Ensure columns exist
    for col in ["order_time", "delivery_time", "Total shipping fee", "tracking_id"]:
        if col not in df_range.columns:
            df_range[col] = pd.NA

    # Parse timestamps for filtering
    order_times = pd.to_datetime(df_range.get("order_time"), utc=True, errors="coerce")
    delivery_times = pd.to_datetime(df_range.get("delivery_time"), utc=True, errors="coerce")
    
    # === KPI Section ===
    st.subheader("📊 区间指标")
    
    # Order count: prefer tracking_id when reliable; otherwise fall back to row count
    use_tracking_id = _use_tracking_id_for_count(df_range)
    tracking_ids = df_range["tracking_id"].astype(str).str.strip()
    tracking_ids_clean = tracking_ids[tracking_ids != ""]
    if use_tracking_id:
        order_count = tracking_ids_clean.nunique()
        order_caption = "COUNT DISTINCT tracking_id"
    else:
        order_count = len(df_range)
        order_caption = "按行计数（tracking_id 不可靠）"
    
    # Revenue sum (only delivery_time non-null)
    delivery_mask = delivery_times.notna()
    fees_numeric = pd.to_numeric(
        df_range.loc[delivery_mask, "Total shipping fee"].astype(str).str.replace("[$,]", "", regex=True).str.strip(),
        errors="coerce"
    )
    revenue_sum = float(fees_numeric.sum(skipna=True)) if not fees_numeric.empty else 0.0
    if pd.isna(revenue_sum):
        revenue_sum = 0.0

    kpi_c1, kpi_c2 = st.columns(2)
    with kpi_c1:
        st.metric(label="订单总数", value=f"{order_count:,}")
        st.caption(order_caption)
    
    with kpi_c2:
        st.metric(label="营收总数", value=_fmt_currency(revenue_sum))
        st.caption("仅统计 delivery_time 非空")

    # === Weekly Charts ===
    st.subheader("📊 每周数据（所选时间范围）")
    
    # Add week/year columns
    df_range["_week"] = order_times.dt.isocalendar().week
    df_range["_year"] = order_times.dt.isocalendar().year
    df_range["_week_label"] = df_range["_year"].astype(str) + "-W" + df_range["_week"].astype(str).str.zfill(2)
    
    # Count unique tracking_id by week
    df_week_calc = df_range.copy()
    df_week_calc["_tr_id"] = df_week_calc["tracking_id"].astype(str).str.strip()
    df_week_nonempty = df_week_calc[df_week_calc["_tr_id"] != ""]
    
    if use_tracking_id:
        if len(df_week_nonempty) > 0:
            weekly_orders_df = df_week_nonempty.groupby("_week_label", sort=False)["tracking_id"].nunique().reset_index(name="order_count")
        else:
            weekly_orders_df = df_range.groupby("_week_label", sort=False).size().reset_index(name="order_count")
    else:
        weekly_orders_df = df_range.groupby("_week_label", sort=False).size().reset_index(name="order_count")
    
    # Left: Weekly Order Volume
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        if len(weekly_orders_df) > 0:
            fig_orders = go.Figure()
            fig_orders.add_trace(go.Bar(
                x=weekly_orders_df["_week_label"],
                y=weekly_orders_df["order_count"],
                name="周订单量",
                marker_color="#2ca02c",
                marker_line_width=0
            ))
            fig_orders.update_layout(
                title="每周订单量",
                xaxis_title="周",
                yaxis_title="订单量",
                hovermode="x unified",
                showlegend=False,
                height=400,
                bargap=0.15
            )
            max_order = weekly_orders_df["order_count"].max()
            fig_orders.update_yaxes(range=[0, max_order * 1.15], tickformat="d")
            st.plotly_chart(fig_orders, width='stretch')
        else:
            st.info("暂无周订单数据")

    # Right: Weekly Revenue
    with right_col:
        delivery_mask_weekly = delivery_times.notna()
        df_weekly_rev = df_range[delivery_mask_weekly].copy()
        
        if len(df_weekly_rev) > 0:
            fees_clean = df_weekly_rev["Total shipping fee"].astype(str).str.replace("[$,]", "", regex=True).str.strip()
            df_weekly_rev["_fee_num"] = pd.to_numeric(fees_clean, errors="coerce").fillna(0.0)
            
            weekly_revenue = df_weekly_rev.groupby("_week_label", sort=False)["_fee_num"].sum().reset_index(name="revenue")
            
            target_weekly = TARGET_REVENUE / 52
            fig_revenue = go.Figure()
            fig_revenue.add_trace(go.Bar(
                x=weekly_revenue["_week_label"],
                y=weekly_revenue["revenue"],
                name="周营收",
                marker_color="#1f77b4",
                marker_line_width=0
            ))
            fig_revenue.add_hline(
                y=target_weekly,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"目标周营收：{_fmt_currency(target_weekly)}",
                annotation_position="right"
            )
            fig_revenue.update_layout(
                title="每周营收（仅已交付）",
                xaxis_title="周",
                yaxis_title="营收（美元）",
                hovermode="x unified",
                showlegend=False,
                height=400,
                bargap=0.15
            )
            max_rev = weekly_revenue["revenue"].max()
            fig_revenue.update_yaxes(range=[0, max(max_rev, target_weekly) * 1.15], tickformat="$,.0f")
            st.plotly_chart(fig_revenue, width='stretch')
        else:
            st.info("暂无已交付订单数据")

    # === 州-州组合订单量柱状图（左半屏） ===
    # 从 df_range 直接读取已有的 state_pair（由 enrich_df_with_states 生成）
    # Drop rows with missing state on either side for aggregation
    df_pairs = df_range.loc[df_range["pickup_state"].notna() & df_range["delivery_state"].notna()].copy()

    if df_pairs.shape[0] == 0:
        left_col, right_col = st.columns([1, 1])
        with left_col:
            st.info("所选时间范围内暂无州-州组合订单数据")
        # right_col intentionally left blank
    else:
        df_pairs["state_pair"] = df_pairs["pickup_state"].astype(str).str.upper() + "-" + df_pairs["delivery_state"].astype(str).str.upper()

        # Count orders per state_pair (prefer tracking_id when reliable)
        use_tracking_id = _use_tracking_id_for_count(df_pairs)
        if use_tracking_id:
            tr = df_pairs["tracking_id"].astype("string").str.strip()
            nonempty_mask = tr.notna() & (tr != "")
            agg = (
                df_pairs[nonempty_mask]
                .groupby("state_pair", sort=False)["tracking_id"]
                .nunique()
                .reset_index(name="order_count")
            )
        else:
            agg = df_pairs.groupby("state_pair", sort=False).size().reset_index(name="order_count")

        if agg.shape[0] == 0:
            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.info("所选时间范围内暂无州-州组合订单数据")
        else:
            agg = agg.sort_values("order_count", ascending=False).reset_index(drop=True)
            if len(agg) > 10:
                topn = agg.iloc[:10].copy()
                others_sum = int(agg.iloc[10:]["order_count"].sum())
                others_row = pd.DataFrame([{"state_pair": "其他", "order_count": others_sum}])
                plot_df = pd.concat([topn, others_row], ignore_index=True)
            else:
                plot_df = agg.copy()

            plot_df["state_pair"] = plot_df["state_pair"].astype(str)
            x = plot_df["state_pair"].tolist()
            y = plot_df["order_count"].astype(int).tolist()
            max_order = plot_df["order_count"].max()
            y_max = max_order * 1.15

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x,
                y=y,
                marker_color="#2ca02c",
                text=y,
                textposition="outside",
                texttemplate="%{text:d}"
            ))
            fig.update_layout(
                title="提货州-送货州 订单量分布（所选时间范围）",
                xaxis_title="州组合",
                yaxis_title="订单量",
                hovermode="x unified",
                showlegend=False,
                height=400,
                uniformtext_minsize=10,
                uniformtext_mode="hide",
            )
            fig.update_yaxes(range=[0, y_max], tickformat="d")

            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.plotly_chart(fig, width="stretch")
            
            # === 州-州营收柱状图（右列） ===
            with right_col:
                # Revenue aggregation (only delivered orders)
                delivery_mask_state = delivery_times.notna()
                df_state_rev = df_pairs[df_pairs.index.isin(df_range[delivery_mask_state].index)].copy()
                
                if len(df_state_rev) > 0:
                    fees_clean = df_state_rev["Total shipping fee"].astype(str).str.replace("[$,]", "", regex=True).str.strip()
                    df_state_rev["_fee_num"] = pd.to_numeric(fees_clean, errors="coerce").fillna(0.0)
                    
                    revenue_agg = df_state_rev.groupby("state_pair", sort=False)["_fee_num"].sum().reset_index(name="revenue")
                    
                    if revenue_agg.shape[0] > 0:
                        revenue_agg = revenue_agg.sort_values("revenue", ascending=False).reset_index(drop=True)
                        if len(revenue_agg) > 10:
                            topn_rev = revenue_agg.iloc[:10].copy()
                            others_rev = float(revenue_agg.iloc[10:]["revenue"].sum())
                            others_row_rev = pd.DataFrame([{"state_pair": "其他", "revenue": others_rev}])
                            plot_df_rev = pd.concat([topn_rev, others_row_rev], ignore_index=True)
                        else:
                            plot_df_rev = revenue_agg.copy()
                        
                        plot_df_rev["state_pair"] = plot_df_rev["state_pair"].astype(str)
                        x_rev = plot_df_rev["state_pair"].tolist()
                        y_rev = plot_df_rev["revenue"].astype(float).tolist()
                        max_rev = plot_df_rev["revenue"].max()
                        y_max_rev = max_rev * 1.15
                        
                        fig_rev = go.Figure()
                        fig_rev.add_trace(go.Bar(x=x_rev, y=y_rev, marker_color="#1f77b4"))
                        fig_rev.update_layout(
                            title="提货州-送货州 营收分布（所选时间范围）",
                            xaxis_title="州组合",
                            yaxis_title="营收（美元）",
                            hovermode="x unified",
                            showlegend=False,
                            height=400,
                        )
                        fig_rev.update_yaxes(range=[0, y_max_rev], tickformat="$,.0f")
                        st.plotly_chart(fig_rev, width="stretch")
                    else:
                        st.info("暂无州-州营收数据")
                else:
                    st.info("暂无已交付的州-州组合数据")

    # === Customer Pies ===
    st.subheader("👥 客户结构（Top 8 + 其他）")
    render_customer_pies_mini(df_range)


def render_customer_pies_mini(df_range: pd.DataFrame):
    """
    客户饼图（订单量 & 营收）
    """
    # Ensure columns
    if "Customer ID" not in df_range.columns:
        st.info("数据中缺少 Customer ID 列")
        return

    if len(df_range) == 0:
        st.info("暂无客户数据")
        return

    # Clean Customer ID
    cust_series = df_range["Customer ID"].astype(str).str.strip()
    valid_cust_mask = df_range["Customer ID"].notna() & (cust_series != "")
    df_valid = df_range[valid_cust_mask].copy()

    if len(df_valid) == 0:
        st.info("暂无有效客户数据")
        return

    # --- Orders pie ---
    tracking_ids = df_valid["tracking_id"].astype(str).str.strip()
    tracking_ids_clean = tracking_ids[tracking_ids != ""]
    use_tracking_id = _use_tracking_id_for_count(df_valid)
    
    if use_tracking_id and len(tracking_ids_clean) > 0:
        orders_agg = df_valid[tracking_ids != ""].groupby("Customer ID", dropna=False)["tracking_id"].nunique().reset_index(name="order_count")
    else:
        orders_agg = df_valid.groupby("Customer ID", dropna=False).size().reset_index(name="order_count")

    orders_agg = orders_agg.sort_values("order_count", ascending=False).reset_index(drop=True)
    
    # Top 8 + Others
    if len(orders_agg) > 8:
        top_orders = orders_agg.iloc[:8].copy()
        others_sum = orders_agg.iloc[8:]["order_count"].sum()
        top_orders = pd.concat([top_orders, pd.DataFrame([{"Customer ID": "其他", "order_count": others_sum}])], ignore_index=True)
    else:
        top_orders = orders_agg.copy()

    top_orders = top_orders.rename(columns={"Customer ID": "Customer", "order_count": "value"})

    # --- Revenue pie (only delivered) ---
    delivery_times = pd.to_datetime(df_valid.get("delivery_time"), utc=True, errors="coerce")
    delivery_mask = delivery_times.notna()
    df_delivered = df_valid[delivery_mask].copy()

    left_col, right_col = st.columns([1, 1])

    # Left: Orders Pie
    with left_col:
        if top_orders["value"].sum() == 0:
            st.info("暂无订单数据")
        else:
            fig_o = px.pie(top_orders, names="Customer", values="value", title="订单量占比")
            fig_o.update_traces(
                textinfo="percent+label",
                hovertemplate="%{label}<br>%{percent:.1%}<br>订单量: %{value:.0f}<extra></extra>"
            )
            fig_o.update_layout(legend=dict(orientation="v", y=0.5, x=1.02))
            st.plotly_chart(fig_o, width='stretch')

    # Right: Revenue Pie
    with right_col:
        if len(df_delivered) > 0:
            fees_clean = df_delivered["Total shipping fee"].astype(str).str.replace("[$,]", "", regex=True).str.strip()
            df_delivered["_fee_num"] = pd.to_numeric(fees_clean, errors="coerce").fillna(0.0)
            
            revenue_agg = df_delivered.groupby("Customer ID", dropna=False)["_fee_num"].sum().reset_index(name="revenue")
            revenue_agg = revenue_agg.sort_values("revenue", ascending=False).reset_index(drop=True)
            
            if len(revenue_agg) > 8:
                top_rev = revenue_agg.iloc[:8].copy()
                others_rev = revenue_agg.iloc[8:]["revenue"].sum()
                top_rev = pd.concat([top_rev, pd.DataFrame([{"Customer ID": "其他", "revenue": others_rev}])], ignore_index=True)
            else:
                top_rev = revenue_agg.copy()

            top_rev = top_rev.rename(columns={"Customer ID": "Customer", "revenue": "value"})
            
            if top_rev["value"].sum() > 0:
                fig_r = px.pie(top_rev, names="Customer", values="value", title="营收占比（仅已交付）")
                fig_r.update_traces(
                    textinfo="percent+label",
                    hovertemplate="%{label}<br>%{percent:.1%}<br>营收: ${value:,.0f}<extra></extra>"
                )
                fig_r.update_layout(legend=dict(orientation="v", y=0.5, x=1.02))
                st.plotly_chart(fig_r, width='stretch')
            else:
                st.info("暂无已交付营收数据")
        else:
            st.info("暂无已交付订单数据")


def render_tab_lead_time(df_range: pd.DataFrame):
    """
    Tab2：时效
    包含：3个KPI + 周平均时效趋势图
    """
    if len(df_range) == 0:
        st.info("所选时间范围内暂无数据")
        return

    # Ensure columns exist
    for col in ["order_time", "delivery_time", "facility_check_in_time"]:
        if col not in df_range.columns:
            df_range[col] = pd.NA

    # Parse timestamps
    order_times = pd.to_datetime(df_range.get("order_time"), utc=True, errors="coerce")
    delivery_times = pd.to_datetime(df_range.get("delivery_time"), utc=True, errors="coerce")
    checkin_times = pd.to_datetime(df_range.get("facility_check_in_time"), utc=True, errors="coerce")

    # === 入库时效分布图（按州-州+Zone维度） ===
    st.subheader("⏱️ 入库时效分布（所选时间范围）")
    
    # 筛选 order_time 和 facility_check_in_time 都非空的订单
    mask_checkin = order_times.notna() & checkin_times.notna()
    
    if mask_checkin.sum() > 0:
        df_checkin = df_range[mask_checkin].copy()
        
        # 计算入库时效（小时）
        df_checkin["_checkin_duration_hours"] = (checkin_times[mask_checkin] - order_times[mask_checkin]).dt.total_seconds() / 3600
        
        # 确保 pickup_state 和 delivery_state 存在（来自前面的 ZIP→State 映射）
        if "pickup_state" not in df_checkin.columns:
            df_checkin["pickup_state"] = None
        if "delivery_state" not in df_checkin.columns:
            df_checkin["delivery_state"] = None
        if "zone" not in df_checkin.columns:
            df_checkin["zone"] = "Unknown"
        
        # 构建 state_pair 和 zone 标签
        df_checkin["state_pair"] = df_checkin["pickup_state"].astype(str) + "-" + df_checkin["delivery_state"].astype(str)
        df_checkin["state_zone_label"] = df_checkin["state_pair"].astype(str) + " | Zone " + df_checkin["zone"].astype(str)
        
        # 按 state_pair + zone 分组，计算平均入库时效和样本量
        checkin_agg = df_checkin.groupby("state_zone_label", sort=False).agg({
            "_checkin_duration_hours": ["mean", "count"]
        }).reset_index()
        checkin_agg.columns = ["state_zone_label", "avg_hours", "count"]
        
        # 过滤样本量 >= 10
        checkin_agg = checkin_agg[checkin_agg["count"] >= 10].copy()
        
        if len(checkin_agg) > 0:
            # 排序：取"最慢 Top 10"（按平均时效降序），然后在绘图前按升序排列
            checkin_agg = checkin_agg.sort_values("avg_hours", ascending=False).reset_index(drop=True)
            if len(checkin_agg) > 10:
                checkin_agg = checkin_agg.iloc[:10].copy()
            
            # 再按升序排列以便绘图（最快在上，最慢在下）
            checkin_agg = checkin_agg.sort_values("avg_hours", ascending=True).reset_index(drop=True)
            
            y_labels = checkin_agg["state_zone_label"].tolist()
            x_values = checkin_agg["avg_hours"].astype(float).tolist()
            
            fig_checkin = go.Figure()
            fig_checkin.add_trace(go.Bar(
                y=y_labels,
                x=x_values,
                orientation="h",
                marker_color="#17becf",
                hovertemplate="%{y}<br>平均入库时效: %{x:.1f} 小时<extra></extra>"
            ))
            fig_checkin.update_layout(
                title="入库时效分布（所选时间范围）",
                xaxis_title="平均入库时效（小时）",
                yaxis_title="州-州 | Zone",
                hovermode="closest",
                showlegend=False,
                height=500,
            )
            st.plotly_chart(fig_checkin, use_container_width=True)
        else:
            st.info("暂无足够样本数据（需要每个组合至少 10 个订单）")
    else:
        st.info("暂无 order_time 和 facility_check_in_time 都非空的数据")

    # === 配送时效分布图（按州-州+Zone维度） ===
    st.subheader("🚚 配送时效分布（所选时间范围）")

    # 筛选 facility_check_in_time 和 delivery_time 都非空的订单
    mask_delivery = checkin_times.notna() & delivery_times.notna()

    if mask_delivery.sum() > 0:
        df_delivery = df_range[mask_delivery].copy()

        # 计算配送时效（小时）
        df_delivery["_delivery_duration_hours"] = (
            delivery_times[mask_delivery] - checkin_times[mask_delivery]
        ).dt.total_seconds() / 3600

        # 丢弃负数时长
        df_delivery = df_delivery[df_delivery["_delivery_duration_hours"] >= 0].copy()

        if len(df_delivery) > 0:
            # 确保 pickup_state 和 delivery_state 存在
            if "pickup_state" not in df_delivery.columns:
                df_delivery["pickup_state"] = None
            if "delivery_state" not in df_delivery.columns:
                df_delivery["delivery_state"] = None
            if "zone" not in df_delivery.columns:
                df_delivery["zone"] = "Unknown"

            # 构建 state_pair 和 zone 标签
            df_delivery["state_pair"] = df_delivery["pickup_state"].astype(str) + "-" + df_delivery["delivery_state"].astype(str)
            df_delivery["state_zone_label"] = df_delivery["state_pair"].astype(str) + " | Zone " + df_delivery["zone"].astype(str)

            # 按 state_pair + zone 分组，计算平均配送时效和样本量
            delivery_agg = df_delivery.groupby("state_zone_label", sort=False).agg({
                "_delivery_duration_hours": ["mean", "count"]
            }).reset_index()
            delivery_agg.columns = ["state_zone_label", "avg_hours", "count"]

            # 过滤样本量 >= 10
            delivery_agg = delivery_agg[delivery_agg["count"] >= 10].copy()

            if len(delivery_agg) > 0:
                # 按时效从短到长排序
                delivery_agg = delivery_agg.sort_values("avg_hours", ascending=True).reset_index(drop=True)

                y_labels = delivery_agg["state_zone_label"].tolist()
                x_values = delivery_agg["avg_hours"].astype(float).tolist()

                fig_delivery = go.Figure()
                fig_delivery.add_trace(go.Bar(
                    y=y_labels,
                    x=x_values,
                    orientation="h",
                    marker_color="#ff7f0e",
                    text=[f"{v:.1f}h" for v in x_values],
                    textposition="outside",
                    texttemplate="%{text}",
                    hovertemplate="%{y}<br>平均配送时效: %{x:.1f} 小时<extra></extra>"
                ))
                fig_delivery.update_layout(
                    title="配送时效分布（所选时间范围）",
                    xaxis_title="平均配送时效（小时）",
                    yaxis_title="州-州 | Zone",
                    hovermode="closest",
                    showlegend=False,
                    height=500,
                    uniformtext_minsize=10,
                    uniformtext_mode="hide",
                )
                st.plotly_chart(fig_delivery, use_container_width=True)
            else:
                st.info("暂无足够样本数据（需要每个组合至少 10 个订单）")
        else:
            st.info("暂无有效配送时效数据（剔除负数后为空）")
    else:
        st.info("暂无 facility_check_in_time 和 delivery_time 都非空的数据")

    # === KPI Section ===
    st.subheader("📊 时效指标")

    # 1) 平均下单到签收时长
    mask1 = delivery_times.notna()
    if mask1.sum() > 0:
        durations1 = delivery_times[mask1] - order_times[mask1]
        avg_duration1 = durations1.mean()
        avg_duration1_str = _fmt_duration_hours(avg_duration1)
    else:
        avg_duration1_str = "-"

    # 2) 平均下单到入仓时长
    mask2 = checkin_times.notna()
    if mask2.sum() > 0:
        durations2 = checkin_times[mask2] - order_times[mask2]
        avg_duration2 = durations2.mean()
        avg_duration2_str = _fmt_duration_hours(avg_duration2)
    else:
        avg_duration2_str = "-"

    # 3) 平均入仓到签收时长
    mask3 = (checkin_times.notna()) & (delivery_times.notna())
    if mask3.sum() > 0:
        durations3 = delivery_times[mask3] - checkin_times[mask3]
        avg_duration3 = durations3.mean()
        avg_duration3_str = _fmt_duration_hours(avg_duration3)
    else:
        avg_duration3_str = "-"

    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
    with kpi_c1:
        st.metric(label="平均下单到签收", value=avg_duration1_str)
        st.caption(f"订单数: {mask1.sum()}")

    with kpi_c2:
        st.metric(label="平均下单到入仓", value=avg_duration2_str)
        st.caption(f"订单数: {mask2.sum()}")

    with kpi_c3:
        st.metric(label="平均入仓到签收", value=avg_duration3_str)
        st.caption(f"订单数: {mask3.sum()}")

    # 已移除“每周平均下单到签收时长（趋势）”

# -----------------
# Main Page Layout
# -----------------
st.title("FIMILE Dashboard")

# NOTE: Load real data from Google Sheet; fallback to demo if secrets or read fails.

# Demo DataFrame (for fallback only)
demo_data = [
    {
        "order_time": "2026-01-05 08:12:00+0000",
        "delivery_time": "2026-01-10 10:00:00+0000",
        "facility_check_in_time": "2026-01-06 14:00:00+0000",
        "Total shipping fee": 1200,
        "tracking_id": "TRACK001",
        "Customer ID": "CUST_A"
    },
    {
        "order_time": "2026-02-01 12:30:00+0000",
        "delivery_time": "",
        "facility_check_in_time": "2026-02-02 10:00:00+0000",
        "Total shipping fee": 800,
        "tracking_id": "TRACK002",
        "Customer ID": "CUST_B"
    },
    {
        "order_time": "2026-02-03 14:21:00+0000",
        "delivery_time": "2026-02-05 09:00:00+0000",
        "facility_check_in_time": "2026-02-04 11:00:00+0000",
        "Total shipping fee": "1500.50",
        "tracking_id": "TRACK003",
        "Customer ID": "CUST_A"
    },
    {
        "order_time": "2026-02-02 09:15:00+0000",
        "delivery_time": None,
        "facility_check_in_time": "2026-02-03 08:00:00+0000",
        "Total shipping fee": "200",
        "tracking_id": "TRACK004",
        "Customer ID": "CUST_C"
    },
]

# Attempt to load real data from Google Sheet
df, load_error = load_sheet_to_df()

if df is None:
    st.warning(f"⚠️ 无法读取 Google Sheet：{load_error}")
    st.info("已切换到 Demo 数据模式（仅用于测试/验证，不是真实数据）")
    df = pd.DataFrame(demo_data)
else:
    st.success(f"✅ 已成功读取 Google Sheet（{len(df)} 行数据）")

# ===== 顶部固定区 =====

# 预处理：添加 state_pair（仅一次，缓存结果）
df_enriched = enrich_df_with_states(df)

# 模块1：KPI 指标
render_module1_kpis(df)

st.divider()

# 全局时间选择器
start_date, end_date, df_range = render_global_date_filter(df_enriched)

st.divider()

# ===== Tabs 区域 =====
tabs = st.tabs(["订单与营收", "时效"])

with tabs[0]:
    render_tab_orders_revenue(df_range)

with tabs[1]:
    render_tab_lead_time(df_range)
