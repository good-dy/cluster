import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
from sklearn.metrics import pairwise_distances_argmin_min

# 데이터 로딩
@st.cache_data
def load_data():
    df = pd.read_csv("Delivery.csv")
    return df

# 엘보우 메서드로 최적 군집 수 찾기
def calculate_elbow(data, max_k=10):
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

# Folium 지도 생성
def generate_folium_map(data, cluster_column):
    center = [data['Latitude'].mean(), data['Longitude'].mean()]
    m = folium.Map(location=center, zoom_start=12)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'gray', 'black', 'pink']

    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=colors[row[cluster_column] % len(colors)],
            fill=True,
            fill_color=colors[row[cluster_column] % len(colors)],
            fill_opacity=0.7,
            popup=f"Num: {row['Num']}, Cluster: {row[cluster_column]}"
        ).add_to(m)

    return m

# Streamlit 앱
def main():
    st.set_page_config(page_title="📦 배달 위치 군집 분석", layout="wide")
    st.title("📦 배달 위치 군집 분석 with KMeans")

    df = load_data()
    coords = df[['Latitude', 'Longitude']]

    st.subheader("데이터 미리보기")
    st.dataframe(df)

    st.subheader("📈 군집 수 결정: Elbow Method")
    inertias = calculate_elbow(coords, max_k=10)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(range(1, 11)), y=inertias, mode='lines+markers'))
    fig_elbow.update_layout(title="엘보우 그래프", xaxis_title="k (군집 수)", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow, use_container_width=True)

    # 기본 추천: 기울기 급격히 꺾이는 지점 추정
    diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    suggested_k = diffs.index(max(diffs)) + 1 + 1  # +1 for index offset, +1 for next k

    st.success(f"추천 군집 수: {suggested_k}")

    k = st.slider("최종 군집 수 선택", min_value=2, max_value=10, value=suggested_k)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(coords)

    st.subheader("📍 Plotly 시각화")
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color=df["Cluster"].astype(str),
                            zoom=10, height=500, mapbox_style="open-street-map",
                            hover_data=["Num"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗺️ Folium 지도")
    folium_map = generate_folium_map(df, 'Cluster')
    st_data = st_folium(folium_map, width=700)

if __name__ == "__main__":
    main()
