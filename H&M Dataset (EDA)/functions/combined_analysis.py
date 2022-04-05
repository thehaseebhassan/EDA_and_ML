# ==============================================================================
# # Import Modules
# ==============================================================================
from collections import Counter, defaultdict
import matplotlib.ticker as mtick
import plotly.graph_objects as go
import numpy as np                      # linear algebra
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import seaborn as sns                   # nice visualisations
import matplotlib.pyplot as plt         # basic visualisation library
import datetime as dt                   # library to opearate on dates
import gc

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# ==============================================================================
# # Fashion News Frequency
# ==============================================================================


def fashion_news_freq(cust, trans):

    cust_ = pd.DataFrame(
        cust, columns=['club_member_status', 'fashion_news_frequency', 'customer_id'])
    trans_ = pd.DataFrame(trans, columns=['customer_id'])

    # Merging
    custran = pd.merge(cust_, trans_, how='right', on='customer_id')
    ct1 = custran.groupby(['fashion_news_frequency'])[
        'customer_id'].count().reset_index()
    ct2 = ct1.sort_values(['customer_id'], ascending=False)

    # Plot
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#f6f5f5')
    sns.barplot(data=ct2, x="fashion_news_frequency",
                y="customer_id", alpha=0.6, edgecolor='k', linewidth=2)
    ttl = ax.set_title('Fashion News Frequency', fontsize=18)
    ttl.set_position([.5, 1.02])
    ax.set_ylabel('No of Cutomers')
    ax.set_xlabel('Fashion News Frequency')

    plt.subplots_adjust(top=0.85, bottom=0.3, left=0.1, right=0.9)
    plt.show()


# ==============================================================================
# # Words Cloud for Descriptions
# ==============================================================================

def desc_wordcloud(art):
    prod_desc = art[art.detail_desc.notnull()].detail_desc.sample(5000).values

    from wordcloud import WordCloud, STOPWORDS

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800,
                          height=800,
                          background_color='white',
                          min_font_size=10,
                          stopwords=stopwords,).generate(' '.join(prod_desc))

    # plot the WordCloud image
    plt.figure(figsize=(6, 6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()

# ==============================================================================
# # Customers and Products
# ==============================================================================


def cust_art(trans, cust):
    trans['price_band'] = pd.qcut(trans['price'], q=4, labels=[
                                  'low', 'medium', 'high', 'very high'])
    trans['price'] = trans['price'].astype('float32')
    trans['sales_channel_id'] = trans['sales_channel_id'].astype('int32')

    # cust['FN'] = cust['FN'].astype('float32')
    cust['Active'] = cust['Active'].astype('float32')
    cust['age'].fillna(0, inplace=True)
    cust['age'] = cust['age'].astype('int32')

    cust_data = cust
    trans_data = trans

    cust_nature = trans_data.groupby(['customer_id', 'price_band'])[
        'article_id'].count().reset_index(name='totalbought')
    cust_price = cust_nature.groupby(
        'price_band')['totalbought'].sum().reset_index(name='totalitems')
    cust_price['perc_share'] = (
        cust_price['totalitems']/cust_price['totalitems'].sum())*100

    # Plot
    import plotly.express as px
    fig = px.pie(cust_price, values='perc_share',
                 names='price_band', title='% Share by Product Price Type')
    fig.show()


def total_transacting(trans, cust):

    # data
    transactions_train = trans
    customers = cust

    # colour_group_name
    transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'])
    transactions_train['year'] = transactions_train['t_dat'].dt.year
    transactions_train['mon'] = transactions_train['t_dat'].dt.month
    transactions_train['day'] = transactions_train['t_dat'].dt.day

    # Total Customers
    customers['age_bucket'] = pd.cut(
        customers['age'], bins=[15, 18, 25, 30, 40, 50, 100])

    # Transacting Customers in each bucket
    a = transactions_train[['customer_id']]
    b = customers[['customer_id', 'age_bucket']]
    c = pd.merge(a, b, how='inner', on='customer_id')
    c = c.groupby(by='age_bucket').agg(
        {'customer_id': 'nunique'}).reset_index()

    # Plot
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(data=customers, x='age_bucket', palette='crest')
    plt.title('Total Customers')
    plt.xlabel('Age Bucket')
    plt.ylabel('Count of Customers')
    plt.subplot(1, 2, 2)
    sns.barplot(data=c, x='age_bucket', y='customer_id', palette='crest')
    plt.title('Transacting Customers')
    plt.xlabel('Age Bucket')
    plt.ylabel('Count of Transacting Customers')
    plt.show()


def articles_stock_price(trans):
    transactions_train = trans
    transactions_train['t_dat'] = pd.to_datetime(
        transactions_train['t_dat'], infer_datetime_format=True)
    transactions_train['year'] = transactions_train['t_dat'].dt.year
    transactions_train['mon'] = transactions_train['t_dat'].dt.month
    transactions_train['day'] = transactions_train['t_dat'].dt.day
    a = transactions_train[['article_id', 'year']]
    b = art[['article_id', 'index_group_name']]
    c = pd.merge(a, b, how='inner')
    d = c.pivot_table(index='year', columns='index_group_name',
                      values='article_id', aggfunc='count')
    d['total'] = d.sum(axis=1)
    d.iloc[:, 0] = np.round((d.iloc[:, 0]/d['total'])*100, 2)
    d.iloc[:, 1] = np.round((d.iloc[:, 1]/d['total'])*100, 2)
    d.iloc[:, 2] = np.round((d.iloc[:, 2]/d['total'])*100, 2)
    d.iloc[:, 3] = np.round((d.iloc[:, 3]/d['total'])*100, 2)
    d.iloc[:, 4] = np.round((d.iloc[:, 4]/d['total'])*100, 2)
    d.drop(['total'], axis=1, inplace=True)

    e = pd.DataFrame(art[['index_group_name']].value_counts())
    e.columns = ['cnt']
    e['pct'] = np.round((e['cnt']/e['cnt'].sum())*100, 2)

    plt.figure(figsize=(25, 6))
    plt.subplot(1, 3, 1)
    sns.heatmap(e[['cnt']], cmap='Blues', annot=True, fmt='d')
    plt.xlabel('Stock Count')
    plt.ylabel('Article group')

    plt.subplot(1, 3, 2)
    sns.heatmap(e[['pct']], cmap='Blues', annot=True, fmt='g')
    plt.xlabel('Stock in Precentage')
    plt.ylabel('Article group')

    plt.subplot(1, 3, 3)
    sns.heatmap(d, annot=True, cmap='Blues', fmt='g')
    plt.title("Year wise Article Group Sales Percentage")
    plt.xlabel('Article group')
    plt.ylabel('Year')
    plt.show()


def top_selling_colors(trans, art):

    transactions_train = trans
    # data
    a = transactions_train[['article_id', 't_dat', 'year', 'mon']]
    b = art[['article_id', 'colour_group_name']]
    c = pd.merge(a, b, how='inner', on=['article_id']).reset_index()

    # color wise count
    return c['colour_group_name'].value_counts()


def top_articles(trans, art):
    # pre-processing
    train_df = trans
    articles_df = art

    train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
    train_df = train_df[["t_dat", "article_id"]]
    monthly_df = train_df.query("'2020-9-1' <= t_dat")
    weekly_df = train_df.query("'2020-9-22' <= t_dat")

    # sales count
    sales_counts = Counter(train_df.article_id)
    for i in range(len(articles_df)):
        articles_df.at[i,
                       "sales_count"] = sales_counts[articles_df.at[i, "article_id"]]

    # monthly
    monthly_sales_counts = Counter(monthly_df.article_id)
    for i in range(len(articles_df)):
        articles_df.at[i, "monthly_sales_count"] = monthly_sales_counts[articles_df.at[i, "article_id"]]

    # weekly
    weekly_sales_counts = Counter(weekly_df.article_id)
    for i in range(len(articles_df)):
        articles_df.at[i, "weekly_sales_count"] = weekly_sales_counts[articles_df.at[i, "article_id"]]

    # Sorting
    articles_df = articles_df.sort_values(by="sales_count", ascending=False)

    return articles_df


def age_group(trans, cust, art):
    # processing
    transactions_df = trans
    transactions_df["t_dat"] = pd.to_datetime(transactions_df["t_dat"])
    transactions_df = transactions_df.query(
        "'2020-9-16' <= t_dat").reset_index()

    customers_df = cust
    articles_df = art

    transactions_n = transactions_df.shape[0]
    customers_n = customers_df.shape[0]
    articles_n = articles_df.shape[0]

    article_index = {articles_df.article_id[i]: i for i in range(articles_n)}
    customer_index = {
        customers_df.customer_id[i]: i for i in range(customers_n)}

    article_transactions = [list() for _ in range(articles_n)]
    customer_transactions = [list() for _ in range(customers_n)]

    for i in range(transactions_n):
        customer = customer_index[transactions_df.at[i, "customer_id"]]
        article = article_index[transactions_df.at[i, "article_id"]]
        customer_transactions[customer].append(i)
        article_transactions[article].append(i)

    # age count
    age_count = customers_df["age"].value_counts().sort_index()
    # pd.DataFrame({"age":age_count.index, "count":age_count.values}).transpose()

    # This code snippet is counting the number of articles for each age number by going through the whole dataset
    # age
    age_group = defaultdict(int)
    group_count = {"age": [], "count": []}
    temp_sum, temp_group = 0, 0
    for age in age_count.index:
        if temp_group == 0:
            temp_group = age
        age_group[age] = temp_group
        temp_sum += age_count[age]
        if temp_sum < 80000 and age < 99:
            continue
        group_count["age"].append(temp_group)
        group_count["count"].append(temp_sum)
        temp_sum, temp_group = 0, 0
    # pd.DataFrame(group_count).transpose()

    # This code snippet is getting the top articles for each group.
    # Top 12 Articles

    cut_age = list(map(int, sorted(set(age_group.values()))))
    cut_age.append(100)

    recommendations = dict()
    for i in range(len(cut_age) - 1):
        group_transactions = []
        query = str(cut_age[i]) + "<= age < " + str(cut_age[i + 1])
        temp_df = customers_df.query(query)
        for j in temp_df.index:
            group_transactions.extend(customer_transactions[j])
        top12 = transactions_df.loc[group_transactions].article_id.value_counts(
        ).index[:12]
        recommendations[cut_age[i]] = top12
        # print(top12)

    return recommendations
