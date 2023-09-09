import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
import pickle
import streamlit as st


# Read data
df = pd.read_csv("CDNOW_master.txt", sep='\s+', header=None, names=['Customer_id', 'day', 'Quantity', 'Sales'])

#--------------
# GUI
st.title("Data Science Project")
st.header("Recomender System",divider='rainbow')

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['txt'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.to_csv("CDNOW_master_new.txt", index = False)

# Clean data
df['day'] = pd.to_datetime(df['day'], format='%Y%m%d')
df = df.dropna()

# Data Pre-processing

# Calculate RFM quartiles

max_date = df['day'].max().date()
Recency = lambda x : (max_date - x.max().date()).days
Frequency  = lambda x: len(x.unique())
Monetary = lambda x : round(sum(x), 2)

df_RFM = df.groupby('Customer_id').agg({'day': Recency,
                                        'Quantity': Frequency,
                                        'Sales': Monetary })
df_RFM.columns = ['Recency','Frequency','Monetary']
df_RFM = df_RFM.sort_values('Monetary', ascending = False)

# Build model Kmeans with k=4
from sklearn.cluster import KMeans

model = KMeans(n_clusters=5, random_state=42)
model.fit(df_RFM)

df_RFM_kmeans = df_RFM.copy()
df_RFM_kmeans["Cluster"] = model.labels_
df_RFM_kmeans.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':['mean', 'count']}).round(2)

cluster_kmeans_count = df_RFM_kmeans['Cluster'].value_counts()

# Calculate average values for each RFM_Level, and return a size of each segment
rfm_agg_kmeans = df_RFM_kmeans.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg_kmeans.columns = rfm_agg_kmeans.columns.droplevel()
rfm_agg_kmeans.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg_kmeans['Percent'] = round((rfm_agg_kmeans['Count']/rfm_agg_kmeans.Count.sum())*100, 2)

# Reset the index
rfm_agg_kmeans = rfm_agg_kmeans.reset_index()

# Change thr Cluster Columns Datatype into discrete values
rfm_agg_kmeans['Cluster'] = 'Cluster '+ rfm_agg_kmeans['Cluster'].astype('str')

#5. Save models
pkl_filename = "customer_segmantation.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)
  

#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    customer_segmantation = pickle.load(file)


# GUI
menu = ["Business Objective", "EDA", "Build Project", "Choose Cluster"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Customer segmentation is a fundamental task in marketing and customer relationship management. With the advancements in data analytics and machine learning, it is now possible to group customers into distinct segments with a high degree of precision, allowing businesses to tailor their marketing strategies and offerings to each segment's unique needs and preferences.
    """)  
    st.write("""###### => Problem/Requirement: Utilize machine learning and data analysis techniques in Python to perform customer segmentation.""")
    st.image("Customer-Segmentation.png")
 


elif choice == 'EDA':    
    st.header("Exploratory Data Analysis")
    st.write('## Sample data')
    st.dataframe(df.sample(5))
    st.text('Transactions timeframe from {} to {}'.format(df['day'].min(), df['day'].max()))
    st.text('{:,} transactions don\'t have a customer id'.format(df[df.Customer_id.isnull()].shape[0]))
    st.text('{:,} unique customer_id'.format(len(df.Customer_id.unique())))

    # Vẽ biểu đồ đường sử dụng matplotlib và seaborn
    df_time = df[['day', 'Sales']]
    df_time['day'] = pd.to_datetime(df_time['day'])
    df_time['month'] = df_time['day'].dt.strftime('%Y-%m')
    monthly_sales = df_time.groupby('month')['Sales'].sum().reset_index()
    fig = px.line(monthly_sales, x='month', y='Sales', title='Doanh số theo từng tháng')
    st.plotly_chart(fig)
    

    temp=pd.DataFrame()
    temp['Year']= pd.DatetimeIndex(df['day']).year
    temp['Sales'] = df.Sales
    group_year = temp.groupby('Year').sum()
    # Vẽ biểu đồ cột
    fig = px.bar(group_year, x=group_year.index, y='Sales', labels={'Year': 'Năm', 'Sales': 'Doanh số'}, title='Doanh số theo năm')
    st.plotly_chart(fig)

    # Hiển thị bảng dữ liệu
    st.write('### Doanh số theo năm')
    st.dataframe(group_year)

elif choice == 'Build Project':
    st.title('Data RFM')
    st.dataframe(df_RFM.head())

    st.title('Biểu đồ Elbow Method')
    sse = {}
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_RFM)
        sse[k] = kmeans.inertia_ # SSE to closest cluster centroid
    # Biểu đồ Elbow Method
    fig, ax = plt.subplots()
    ax.set_title('The Elbow Method')
    ax.set_xlabel('k')
    ax.set_ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()), ax=ax)
    st.pyplot(fig)

    #Data dùng kmeans có cluster
    st.title('Prediction using Kmeans with k=5')
    st.dataframe(df_RFM_kmeans.sample(5))
    st.write(cluster_kmeans_count)

    st.dataframe(rfm_agg_kmeans)
    st.write('''
Cluster 0: "Regulars"\n
Cluster 1: "New"\n
Cluster 2: "Stars"\n
Cluster 3: "Lost"\n
Cluster 4: "Royal" ''')

    fig = px.scatter(
        rfm_agg_kmeans,
        x="RecencyMean",
        y="MonetaryMean",
        size="FrequencyMean",
        color="Cluster",
        log_x=True,
        size_max=60,)
    
    st.plotly_chart(fig, theme=None, use_container_width=True)



elif choice == 'Choose Cluster':
    st.title('Biểu đồ Elbow Method')
    sse = {}
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_RFM)
        sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

    # Biểu đồ Elbow Method
    fig, ax = plt.subplots()
    ax.set_title('The Elbow Method')
    ax.set_xlabel('k')
    ax.set_ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()), ax=ax)
    st.pyplot(fig)

    #Chọn k
    n_clusters = st.sidebar.number_input('Chọn số cụm muốn phân loại từ 2 - 20', min_value=2, max_value=20, step=1, key="cluster_value")
    st.write('Số cụm muốn phân loại là ', n_clusters)

    model = KMeans(n_clusters= n_clusters, random_state=42)
    model.fit(df_RFM)

    df_sub = df_RFM.copy()
    df_sub["Cluster"] = model.labels_
    df_sub.groupby('Cluster').agg({
        'Recency':'mean',
        'Frequency':'mean',
        'Monetary':['mean', 'count']}).round(2)

    cluster_df_sub_count = df_sub['Cluster'].value_counts()

    # Calculate average values for each RFM_Level, and return a size of each segment
    rfm_agg_df_sub = df_sub.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg_df_sub.columns = rfm_agg_df_sub.columns.droplevel()
    rfm_agg_df_sub.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg_df_sub['Percent'] = round((rfm_agg_df_sub['Count']/rfm_agg_df_sub.Count.sum())*100, 2)

    # Reset the index
    rfm_agg_df_sub = rfm_agg_df_sub.reset_index()

    # Change thr Cluster Columns Datatype into discrete values
    rfm_agg_df_sub['Cluster'] = 'Cluster '+ rfm_agg_df_sub['Cluster'].astype('str')

    st.write('Prediction using Kmeans with k= ', n_clusters)
    st.write(cluster_df_sub_count)

    st.dataframe(rfm_agg_df_sub)

    fig = px.scatter(
        rfm_agg_df_sub,
        x="RecencyMean",
        y="MonetaryMean",
        size="FrequencyMean",
        color="Cluster",
        log_x=True,
        size_max=60,)
    
    st.plotly_chart(fig, theme=None, use_container_width=True)

