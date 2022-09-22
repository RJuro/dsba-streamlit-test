import streamlit as st
import altair as alt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('https://sds-aau.github.io/SDS-master/M1/data/cities.csv')
data_to_cluster = data.iloc[:,4:]
scaler = StandardScaler()
data_to_cluster_scaled = scaler.fit_transform(data_to_cluster)
pca = PCA(n_components=2)
data_reduced_pca = pca.fit_transform(data_to_cluster_scaled)



vis_data = pd.DataFrame(data_reduced_pca)
vis_data['place'] = data['place']
vis_data['country'] = data['alpha-2']
vis_data.columns = ['x', 'y', 'place', 'country']

our_chart = alt.Chart(vis_data).mark_circle(size=60).encode(
    x='x',
    y='y',
    tooltip=['place', 'country']
).interactive()



st.write('My new amazing app and we all want to go home')
st.altair_chart(our_chart, use_container_width=True)
