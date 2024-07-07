import streamlit as st

import folium
from streamlit_folium import st_folium

import pandas as pd
import numpy as np
import time

df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.title("Пример")

st.write("Демонстрация работы веб-приложения")

import pandas as pd
import numpy as np

import geopandas as gpd

from scipy.cluster.vq import kmeans2, vq

from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from shapely import Point, Polygon

import warnings
warnings.filterwarnings('ignore')

def to_azimuth(point, dist, angle):
    lon = point.x * np.pi / 180
    lat = point.y * np.pi / 180
    angle = angle * np.pi / 180
    # angle = np.pi / 2.0 - angle
    angd = dist/6371000
    lat_r = np.arcsin(np.sin(lat) * np.cos(angd) + np.cos(lat) * np.sin(angd) * np.cos(angle))
    a = np.sin(angle) * np.sin(angd) * np.cos(lat)
    b = np.cos(angd) - np.sin(lat) * np.sin(lat_r)
    lon_r = lon + np.arctan2(a, b)
    return Point(np.degrees(lon_r), np.degrees(lat_r))

def to_poligon(point, azimuth):
    return Polygon([point, to_azimuth(point, 200, azimuth+15), to_azimuth(point, 200, azimuth-15)])

def prepare_df(df):
    df_temp = df.explode('points', ignore_index=True)
    df_temp = pd.concat([df_temp[['hash', 'value', 'points']], pd.json_normalize(df_temp['targetAudience'])], axis=1)
    points_temp = pd.json_normalize(df_temp['points'])
    df_temp['points'] = df_temp['points'].astype(str)
    points_temp.index = df_temp['points']

    points_temp = points_temp[~points_temp.index.duplicated(keep='first')]
    points_temp['place_coordinates'] = points_temp.apply(lambda x: Point(x.lon, x.lat), axis=1)
    points_temp['coordinates'] = points_temp.apply(lambda x: to_poligon(x.place_coordinates, x.azimuth), axis=1)

    return df_temp, points_temp

df = pd.read_json('content/train_data.json')
df, points = prepare_df(df)
print(len(df), len(points))

# Преобразуем координаты в метры, используя пулковскую проекцию epsg:2584
points_xy = gpd.GeoDataFrame(points).set_geometry('place_coordinates').set_crs(epsg=4326).to_crs(epsg=2584).geometry
points_xy = points_xy.map(lambda x: x.coords[0]).apply(pd.Series)
points_xy.columns = ['x', 'y']
points['x'] = points_xy['x']
points['y'] = points_xy['y']
points = points[['x', 'y', 'azimuth', 'place_coordinates', 'coordinates']]
del points_xy

df_xy = pd.merge(df, points, left_on='points', right_index=True)[['points', 'x', 'y']]

# Возьмем центроид и отсечем лишнее (>100 км от центроида)
x, y = df_xy.x.mean(), df_xy.y.mean()
df_xy['r'] = np.sqrt((df_xy.x-x)*(df_xy.x-x)+(df_xy.y-y)*(df_xy.y-y))

df = df[df_xy.r <=100000]

points = points.iloc[points.index.isin(set(df.points))]

df_xy = df_xy[df_xy.r <=100000]

# Кластеризация

z = list(zip(df_xy.x,df_xy.y))

n_clusters = 1000
centroids, label = kmeans2(z, n_clusters, minit='points')

clusters, dist = vq(z, centroids)
print(f"Кластеров: {len(set(clusters))}\nСреднее расстояние: {dist.mean():.1f}")

points["mean_value"] = df.groupby("points")["value"].mean().astype(int)
points["qty"] = df.groupby("points")["value"].count()


def create_config(df_records, n_clusters):
    return kmeans2(df_records[['x','y']], n_clusters, minit='points')

def create_dataset(config, df_campaign, df_records):
    centroids, _ = config
    clusters, dist = vq(df_records[['x','y']], centroids)
    df_dataset = df_campaign[['hash']]
    df_temp = df_records[['hash']]
    df_temp['claster'] = clusters
    df_temp = df_temp.groupby(["hash", "claster"])["hash"].count().unstack(fill_value=0)
    for c in set(range(len(centroids)))-set(df_temp.columns):
        df_temp[c] = 0
    return pd.merge(df_dataset, df_temp.sort_index(axis=1).rename_axis(None, axis=1).reset_index(), on='hash').set_index('hash')

def to_meters(points):
    points_xy = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326").to_crs(epsg=2584).geometry
    points_xy = points_xy.map(lambda x: x.coords[0]).apply(pd.Series)
    return(points_xy.values[:,0], points_xy.values[:,1])

def from_meters(x, y):
    return gpd.GeoDataFrame(geometry=pd.Series(zip(x, y)).apply(Point), crs="EPSG:2584").to_crs(epsg=4326).values


def prepare_df(_df, centroid=None):
    # собираем данные по кампаниям
    _df_campaigns = pd.json_normalize(_df['targetAudience'])
    _df_campaigns['hash'] = _df['hash'].reset_index(drop=True)

    # вытаскиваем координаты стендов и параметры рекламных кампаний
    _df_temp = _df.explode('points', ignore_index=True)
    #     df_temp = pd.concat([df_temp[['hash', 'points']], pd.json_normalize(df_temp['targetAudience'])], axis=1)

    _points_temp = pd.json_normalize(_df_temp['points'])
    _df_temp['points'] = _df_temp['points'].astype(str)
    _points_temp.index = _df_temp['points']
    _points_temp = _points_temp[~_points_temp.index.duplicated(keep='first')]

    # кооординаты стенда - point coords
    _points_temp['p_coords'] = _points_temp.apply(lambda x: Point(x.lon, x.lat), axis=1)
    _points_temp['x'], _points_temp['y'] = to_meters(_points_temp.p_coords)

    # полигон ожидаемого охвата территории стендом с учетом азимута - coverage poligons
    _points_temp['c_coords'] = _points_temp.apply(lambda x: to_poligon(x.p_coords, x.azimuth), axis=1)

    _df_temp = pd.merge(_df_temp, _points_temp[['x', 'y', 'azimuth']], left_on='points', right_index=True)

    # расчет расстояний от точек размещения до центроида всей выборки стендов в метрах
    # определение центроида, если трен
    if centroid is None:
        centroid = kmeans2(_df_temp[['x', 'y']], 1)[0]
    # определение расстояния до центроида
    _df_temp['r'] = vq(_df_temp[['x', 'y']], centroid)[1]

    return (_df_campaigns[['hash', 'name', 'gender', 'ageFrom', 'ageTo', 'income']],
            _df_temp[['hash', 'points', 'x', 'y', 'azimuth', 'r']],
            _points_temp[['p_coords', 'c_coords']], centroid)

df = pd.read_json('content/train_data.json')
X, y = df[['hash','targetAudience','points']], df['value']

df_campaign, df_records, df_points, centroid = prepare_df(df)

df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_campaign_train = df_campaign.iloc[df_train.index]
df_records_train = df_records[df_records['hash'].isin(set(df_campaign_train.hash))]
df_points_train = df_points[df_points.index.isin(set(df_records_train.points))]

df_campaign_test = df_campaign.iloc[df_test.index]
df_records_test = df_records[df_records['hash'].isin(set(df_campaign_test.hash))]
df_points_test = df_points[df_points.index.isin(set(df_records_test.points))]

n_groups = 1000
config = create_config(df_records_train, n_groups)
X_train = create_dataset(config, df_campaign_train, df_records_train)
X_train = X_train.reset_index(drop=True).set_index(y_train.index)
X_test = create_dataset(config, df_campaign_test, df_records_test)
X_test = X_test.reset_index(drop=True).set_index(y_test.index)

model = CatBoostRegressor(iterations=1200,
                          depth=8,
                          learning_rate=0.05,
                          loss_function='RMSE')

model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Метрика при n_groups={n_groups}: {max(1 - rmse/30, 0)**4:.4f}')

# https://www.openstreetmap.org/- информация по poi (светофоры, развязки, ТЦ)
# https://dtp-stat.ru/opendata - информация по ДТП

model = RandomForestRegressor(n_estimators=200, verbose=0, random_state=42, criterion= "friedman_mse")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Метрика при n_groups={n_groups}: {max(1 - rmse/30, 0)**4:.4f}')

feat_importances = pd.Series(model.feature_importances_, index=X_test.columns)

POI = feat_importances.nlargest(10).index

POI_points = config[0][POI]

POI_points_geometry = from_meters(POI_points[:, 0], POI_points[:, 1])

m = folium.Map(location=(55.755826, 37.6173), zoom_start=9, attributionControl=0)
folium.TileLayer('CartoDB positron', name = 'CartoDB positron').add_to(m)
folium.TileLayer('https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                 attr = 'Google Satellite', name = 'Google Спутник').add_to(m)
folium.TileLayer('http://mt0.google.com/vt/lyrs=y&hl=ru&x={x}&y={y}&z={z}',
                 attr = 'Google Hybrid', name = 'Google Гибрид').add_to(m)
folium.TileLayer('https://mt0.google.com/vt/lyrs=m&hl=ru&x={x}&y={y}&z={z}',
                 attr = 'Google Roadmap', name = 'Google Схема').add_to(m)

fg = folium.FeatureGroup(name='>1 кампании', show=True)
m.add_child(fg)
gdf_points = gpd.GeoDataFrame(points[points['qty']>1][['mean_value', 'qty', 'coordinates']])
gdf_points.set_geometry('coordinates', inplace=True)
gdf_points.set_crs(epsg=4326, inplace=True)
gdf_points['color'] = gdf_points.mean_value.map(lambda x : np.log10(x+1)/np.log10(100))
gdf_points.explore(column='color', cmap='winter', m=fg, legend=False) # RdYlGn

fg = folium.FeatureGroup(name='Успешные кампании', show=False)
m.add_child(fg)
gdf_points = gpd.GeoDataFrame(points[points['mean_value']>30][['mean_value', 'qty', 'coordinates']])
gdf_points.set_geometry('coordinates', inplace=True)
gdf_points.set_crs(epsg=4326, inplace=True)
gdf_points['color'] = gdf_points.mean_value.map(lambda x : 'red' if x<=10 else 'blue' if x<30 else 'green')
gdf_points.explore(column='color', cmap=['green'], m=fg, legend=False)

fg = folium.FeatureGroup(name='Значимые точки', show=True)
m.add_child(fg)
gdf_points = gpd.GeoDataFrame(POI_points_geometry)
gdf_points.set_geometry(0, inplace=True)
gdf_points.set_crs(epsg=4326, inplace=True)
gdf_points['Значимость'] = feat_importances.nlargest(10).values

gdf_points.explore(color = 'red', m=fg, legend=False)

m.add_child(folium.LayerControl())
st_data = st_folium(m, width=725)