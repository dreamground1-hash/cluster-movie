import streamlit as st
import pandas as pd
import plotly.express as px
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 페이지 설정
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 특성을 바탕으로 K-평균 군집화를 수행합니다.")

# 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    numeric_cols = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["movieNm"] = df["movieNm"].fillna("제목 없음").astype(str)

    # 롱런 지수 계산
    df["long_run_index"] = (
        df["total_audi"] / df["first_week_audi"]
    ).where(df["first_week_audi"] > 0)

    # 롱런 지수가 20을 넘으면 20으로 제한
    df["long_run_index"] = df["long_run_index"].clip(upper=20)

    return df


try:
    original = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 데이터 주소와 인터넷 연결을 확인해 주세요.")
    st.exception(e)
    st.stop()

# 결측치와 첫 주 관객이 0인 영화 제외
required_cols = [
    "first_scrn",
    "total_audi",
    "days_in_top10",
    "long_run_index",
    "first_week_audi"
]

valid = original.dropna(subset=required_cols).copy()

valid = valid[
    (valid["first_week_audi"] > 0)
    & (valid["first_scrn"] >= 0)
    & (valid["total_audi"] >= 0)
    & (valid["days_in_top10"] >= 0)
].copy()

# 로그 변환
# 값이 0인 경우도 처리하기 위해 log10(값 + 1) 사용
valid["스크린 수(로그)"] = valid["first_scrn"].map(
    lambda x: math.log10(x + 1)
)

valid["누적 관객(로그)"] = valid["total_audi"].map(
    lambda x: math.log10(x + 1)
)

valid["10위권 일수"] = valid["days_in_top10"]
valid["롱런 지수"] = valid["long_run_index"]

FEATURES = [
    "스크린 수(로그)",
    "누적 관객(로그)",
    "10위권 일수",
    "롱런 지수"
]

DISPLAY_NAMES = {
    "스크린 수(로그)": "스크린 수 (log10)",
    "누적 관객(로그)": "누적 관객 (log10)",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수"
}

# 전체 편수와 분석 대상 편수
st.write(
    f"**전체 영화 {len(original):,}편 · "
    f"분석에 묶은 영화 {len(valid):,}편**"
)

if len(valid) < 3:
    st.warning("유효한 영화가 3편 미만이라 세 묶음으로 나눌 수 없습니다.")
    st.stop()

# 군집화 속성 선택
st.subheader("1. 군집화 속성 선택")

selected_features = st.multiselect(
    "유형을 나누는 데 사용할 속성을 선택하세요. (2개 이상)",
    options=FEATURES,
    default=FEATURES,
    format_func=lambda x: DISPLAY_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("군집화를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# 표준화 및 K-평균 군집화
X = valid[selected_features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

valid["군집 번호"] = model.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    valid.groupby("군집 번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

label_map = {
    cluster_id: ["㉮", "㉯", "㉰"][i]
    for i, cluster_id in enumerate(cluster_order)
}

valid["유형"] = valid["군집 번호"].map(label_map)

valid["유형"] = pd.Categorical(
    valid["유형"],
    categories=["㉮", "㉯", "㉰"],
    ordered=True
)

# 2차원 산점도
st.subheader("2. 2차원 산점도")

col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox(
        "가로축 속성",
        selected_features,
        index=0,
        format_func=lambda x: DISPLAY_NAMES[x],
        key="scatter_x"
    )

with col2:
    y_axis = st.selectbox(
        "세로축 속성",
        selected_features,
        index=1,
        format_func=lambda x: DISPLAY_NAMES[x],
        key="scatter_y"
    )

fig2d = px.scatter(
    valid,
    x=x_axis,
    y=y_axis,
    color="유형",
    hover_name="movieNm",
    hover_data={
        "유형": True,
        "군집 번호": False
    },
    labels={
        x_axis: DISPLAY_NAMES[x_axis],
        y_axis: DISPLAY_NAMES[y_axis],
        "유형": "영화 유형"
    },
    category_orders={
        "유형": ["㉮", "㉯", "㉰"]
    },
    title="속성에 따른 영화 유형 분포"
)

fig2d.update_traces(
    marker=dict(size=8, opacity=0.8)
)

fig2d.update_layout(
    legend_title_text="영화 유형"
)

st.plotly_chart(fig2d, use_container_width=True)

# 3차원 산점도
st.subheader("3. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 군집화 속성을 "
        "세 개 이상 선택해 주세요."
    )
else:
    col1, col2, col3 = st.columns(3)

    with col1:
        x3 = st.selectbox(
            "X축 속성",
            selected_features,
            index=0,
            format_func=lambda x: DISPLAY_NAMES[x],
            key="x3"
        )

    with col2:
        y3 = st.selectbox(
            "Y축 속성",
            selected_features,
            index=1,
            format_func=lambda x: DISPLAY_NAMES[x],
            key="y3"
        )

    with col3:
        z3 = st.selectbox(
            "Z축 속성",
            selected_features,
            index=2,
            format_func=lambda x: DISPLAY_NAMES[x],
            key="z3"
        )

    fig3d = px.scatter_3d(
        valid,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        hover_name="movieNm",
        hover_data={
            "유형": True,
            "군집 번호": False
        },
        labels={
            x3: DISPLAY_NAMES[x3],
            y3: DISPLAY_NAMES[y3],
            z3: DISPLAY_NAMES[z3],
            "유형": "영화 유형"
        },
        category_orders={
            "유형": ["㉮", "㉯", "㉰"]
        },
        title="세 속성으로 살펴보는 영화 유형"
    )

    fig3d.update_traces(
        marker=dict(size=3, opacity=0.8)
    )

    fig3d.update_layout(
        legend_title_text="영화 유형",
        scene=dict(aspectmode="cube")
    )

    st.plotly_chart(fig3d, use_container_width=True)

# 유형별 평균 및 편수
st.subheader("4. 유형별 요약")

summary = valid.groupby(
    "유형", observed=False
).agg(
    편수=("movieCd", "size"),
    스크린_수_평균=("first_scrn", "mean"),
    누적_관객_평균=("total_audi", "mean"),
    십위권_일수_평균=("days_in_top10", "mean"),
    롱런_지수_평균=("long_run_index", "mean")
).reset_index()

summary = summary.rename(columns={
    "유형": "영화 유형",
    "스크린_수_평균": "스크린 수 평균(개)",
    "누적_관객_평균": "누적 관객 평균(명)",
    "십위권_일수_평균": "10위권 일수 평균(일)",
    "롱런_지수_평균": "롱런 지수 평균"
})

summary["편수"] = summary["편수"].astype(int)

average_cols = [
    "스크린 수 평균(개)",
    "누적 관객 평균(명)",
    "10위권 일수 평균(일)",
    "롱런 지수 평균"
]

summary[average_cols] = summary[average_cols].round(2)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

# 유형별 누적 관객 TOP 5
st.subheader("5. 유형별 누적 관객 TOP 5")

for label in ["㉮", "㉯", "㉰"]:
    group = valid[
        valid["유형"] == label
    ].sort_values(
        "total_audi",
        ascending=False
    ).head(5)

    count = int((valid["유형"] == label).sum())

    with st.expander(
        f"유형 {label} · {count:,}편",
        expanded=True
    ):
        if group.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            top5 = group[
                ["movieNm", "total_audi"]
            ].rename(columns={
                "movieNm": "영화 제목",
                "total_audi": "누적 관객(명)"
            }).reset_index(drop=True)

            top5.index = top5.index + 1

            st.dataframe(
                top5,
                use_container_width=True
            )
