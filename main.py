import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from sklearn.cluster import KMeans
from streamlit_folium import st_folium

# ----------------------------
# 1. 데이터 로딩
# ----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Delivery.csv")
    return df
# ----------------------------
# 2. Elbow Method (SSE)
# ----------------------------
def elbow(X, n):
    sse = []
    for i in range(1, n+1):
        km = KMeans(n_clusters=i, n_init=10, random_state=42)
        km.fit(X)
        sse.append(km.inertia_)
    return sse

# ----------------------------
# 3. 추천 군집 수 결정
# ----------------------------
def recommend_k(inertias):
    diffs = np.diff(inertias)
    if len(diffs) < 2:
        return 3
    slopes = np.diff(diffs)
    knee = np.argmax(slopes) + 2  # +2 because index shift and k starts at 1
    return max(3, min(knee, 10))  # 보정

# ----------------------------
# 4. Folium 지도 생성 함수
# ----------------------------
def generate_folium_map(data, cluster_column, centroids):
    center = [data['Latitude'].mean(), data['Longitude'].mean()]
    m = folium.Map(location=center, zoom_start=12)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightblue', 'gray', 'black', 'pink']

    for _, row in data.iterrows():
        cluster_idx = int(row[cluster_column])
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=colors[cluster_idx % len(colors)],
            fill=True,
            fill_color=colors[cluster_idx % len(colors)],
            fill_opacity=0.7,
            popup=f"Num: {row['Num']}, Cluster: {cluster_idx}"
        ).add_to(m)

    # 클러스터 중심 마커 추가
    for idx, center in enumerate(centroids):
        folium.Marker(
            location=[center[0], center[1]],
            popup=f"Cluster {idx} Center",
            icon=folium.Icon(color="black", icon="info-sign")
        ).add_to(m)

    return m

# ----------------------------
# 5. 메인 앱
# ----------------------------
def main():
    st.set_page_config(page_title="📦 배달 군집 분석", layout="wide")
    st.title("📦 배달 위치 클러스터링 분석")

    df = load_data()
    coords = df[['Latitude', 'Longitude']]

    st.subheader("🧾 데이터 미리보기")
    st.dataframe(df)

    # Elbow 계산 및 시각화
    st.subheader("📈 Elbow Method로 군집 수 추천")
    inertias = elbow(coords, n=10)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(range(1, 11)), y=inertias, mode='lines+markers'))
    fig_elbow.update_layout(title="Elbow Graph (SSE vs K)", xaxis_title="군집 수 (k)", yaxis_title="SSE")
    st.plotly_chart(fig_elbow, use_container_width=True)

    suggested_k = recommend_k(inertias)
    st.success(f"추천 군집 수: {suggested_k}")

    k = st.slider("최종 군집 수 선택", min_value=2, max_value=10, value=suggested_k)

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    df['Cluster'] = model.fit_predict(coords)
    centroids = model.cluster_centers_

    # Plotly 지도
    st.subheader("📍 Plotly 지도 시각화")
    fig_map = px.scatter_mapbox(df, lat="Latitude", lon="Longitude",
                                color=df["Cluster"].astype(str),
                                zoom=10, height=500, mapbox_style="open-street-map",
                                hover_data=["Num"])
    # 중심점 추가
    centroid_df = pd.DataFrame(centroids, columns=["Latitude", "Longitude"])
    centroid_df["Cluster"] = [f"Center {i}" for i in range(len(centroids))]

    fig_map.add_trace(
        px.scatter_mapbox(centroid_df, lat="Latitude", lon="Longitude", text="Cluster",
                          marker=dict(size=14, color="black")).data[0]
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Folium 지도
    st.subheader("🗺️ Folium 지도 시각화")
    folium_map = generate_folium_map(df, 'Cluster', centroids)
    st_data = st_folium(folium_map, width=700, height=500)

    # 군집별 배달 수 막대그래프
    st.subheader("📊 각 군집별 배달 건수 총합")
    grouped = df.groupby("Cluster")["Num"].sum().reset_index()

    fig_bar = px.bar(grouped, x="Cluster", y="Num",
                     labels={"Cluster": "클러스터", "Num": "총 배달 건수"},
                     text_auto=True,
                     title="군집별 배달 건수 총합")
    st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
