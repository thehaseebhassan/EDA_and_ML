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
# # Helping Functions for Articles dataframe
# ==============================================================================


def articles_key_features(art):

    # Articles Key Features

    data = art
    art_dtypes = art.dtypes.value_counts()

    fig = plt.figure(figsize=(5, 2), facecolor='white')

    ax0 = fig.add_subplot(1, 1, 1)
    font = 'monospace'
    ax0.text(1, 0.8, "Key figures", color='black', fontsize=28,
             fontweight='bold', fontfamily=font, ha='center')

    ax0.text(0, 0.4, "{:,d}".format(
        data.shape[0]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
    ax0.text(0, 0.001, "# of rows \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily=font, ha='center')

    ax0.text(0.6, 0.4, "{}".format(
        data.shape[1]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
    ax0.text(0.6, 0.001, "# of features \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily=font, ha='center')

    ax0.text(1.2, 0.4, "{}".format(
        art_dtypes[0]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
    ax0.text(1.2, 0.001, "# of text columns \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily=font, ha='center')

    ax0.text(1.9, 0.4, "{}".format(
        art_dtypes[1]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
    ax0.text(1.9, 0.001, "# of numeric columns \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily=font, ha='center')

    ax0.set_yticklabels('')
    ax0.tick_params(axis='y', length=0)
    ax0.tick_params(axis='x', length=0)
    ax0.set_xticklabels('')

    for direction in ['top', 'right', 'left', 'bottom']:
        ax0.spines[direction].set_visible(False)

    fig.subplots_adjust(top=0.9, bottom=0.2, left=0, hspace=1)

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('#8c8c8c')
    fig.patch.set_facecolor('#f6f6f6')
    ax0.set_facecolor('#f6f6f6')

    plt.show()


def articles_bar_plots(database, col, figsize=(13, 5), pct=False, label='articles'):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#f6f6f6')
    for loc in ['bottom', 'left']:
        ax.spines[loc].set_visible(True)
        ax.spines[loc].set_linewidth(2)
        ax.spines[loc].set_color('black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if pct:
        data = database[col].value_counts()
        data = data.div(data.sum()).mul(100)
        data = data.reset_index()
        ax = sns.barplot(data=data, x=col, y='index',
                         color='#2693d7', lw=1.5, ec='black', zorder=2)
        ax.set_xlabel('% of ' + label, fontsize=10, weight='bold')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    else:
        data = database[col].value_counts().reset_index()
        ax = sns.barplot(data=data, x=col, y='index',
                         color='#2693d7', lw=1.5, ec='black', zorder=2)
        ax.set_xlabel('# of articles' + label)

    ax.grid(zorder=0)
    ax.text(0, -0.75, col, color='black', fontsize=10, ha='left',
            va='bottom', weight='bold', style='italic')
    ax.set_ylabel('')

    plt.show()


# ==============================================================================
# # Helping Functions for Customer dataframe
# ==============================================================================

def customers_key_features(cust):

    mpl.rcParams.update(mpl.rcParamsDefault)
    cust_dtypes = cust.dtypes.value_counts()
    data = cust

    fig = plt.figure(figsize=(5, 2), facecolor='white')

    ax0 = fig.add_subplot(1, 1, 1)
    ax0.text(1.0, 1, "Key figures", color='black', fontsize=28,
             fontweight='bold', fontfamily='monospace', ha='center')

    ax0.text(0, 0.4, "{:,d}".format(
        data.shape[0]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
    ax0.text(0, 0.001, "# of rows \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily='monospace', ha='center')

    ax0.text(0.6, 0.4, "{}".format(
        data.shape[1]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
    ax0.text(0.6, 0.001, "# of features \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily='monospace', ha='center')

    ax0.text(1.2, 0.4, "{}".format(
        cust_dtypes[0]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
    ax0.text(1.2, 0.001, "# of text columns \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily='monospace', ha='center')

    ax0.text(1.9, 0.4, "{}".format(
        cust_dtypes[1]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
    ax0.text(1.9, 0.001, "# of numeric columns \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily='monospace', ha='center')

    ax0.set_yticklabels('')
    ax0.tick_params(axis='y', length=0)
    ax0.tick_params(axis='x', length=0)
    ax0.set_xticklabels('')

    for direction in ['top', 'right', 'left', 'bottom']:
        ax0.spines[direction].set_visible(False)

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('#8c8c8c')
    fig.patch.set_facecolor('#f6f6f6')
    ax0.set_facecolor('#f6f6f6')

    plt.show()


def customers_age(cust):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    ax = sns.histplot(data=cust, x='age',
                      bins=cust['age'].nunique(), stat="percent")
    ax.set_xlabel('Distribution of the customers age')
    for loc in ['bottom', 'left']:
        ax.spines[loc].set_visible(True)
        ax.spines[loc].set_linewidth(2)
        ax.spines[loc].set_color('black')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    median = cust['age'].median()
    ax.axvline(x=median, color="orange", ls="--")
    ax.text(median, 4, 'median: {}'.format(round(median, 1)),
            rotation='horizontal', ha='left', color='grey')
    ax.text(25, 5, 'Distribution of customers age', color='grey',
            fontsize=10, ha='left', va='bottom', weight='bold', style='italic')
    plt.show()


# ==============================================================================
# # Helping Functions for Transactions dataframe
# ==============================================================================


def transactions_key_features(trans):

    mpl.rcParams.update(mpl.rcParamsDefault)

    trans_dtypes = trans.dtypes.value_counts()
    data = trans

    fig = plt.figure(figsize=(5, 2), facecolor='white')

    ax0 = fig.add_subplot(1, 1, 1)
    ax0.text(1.0, 1, "Key figures", color='black', fontsize=28,
             fontweight='bold', fontfamily='monospace', ha='center')

    ax0.text(0, 0.4, "{:,d}".format(
        data.shape[0]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
    ax0.text(0, 0.001, "# of rows \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily='monospace', ha='center')

    ax0.text(0.6, 0.4, "{}".format(
        data.shape[1]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
    ax0.text(0.6, 0.001, "# of features \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily='monospace', ha='center')

    ax0.text(1.2, 0.4, "{}".format(
        trans_dtypes[0]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
    ax0.text(1.2, 0.001, "# of text columns \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily='monospace', ha='center')

    ax0.text(1.9, 0.4, "{}".format(
        trans_dtypes[1]), color='gold', fontsize=24, fontweight='bold', fontfamily='monospace', ha='center')
    ax0.text(1.9, 0.001, "# of numeric columns \nin the dataset", color='dimgrey',
             fontsize=15, fontweight='light', fontfamily='monospace', ha='center')

    ax0.set_yticklabels('')
    ax0.tick_params(axis='y', length=0)
    ax0.tick_params(axis='x', length=0)
    ax0.set_xticklabels('')

    for direction in ['top', 'right', 'left', 'bottom']:
        ax0.spines[direction].set_visible(False)

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('#8c8c8c')
    fig.patch.set_facecolor('#f6f6f6')
    ax0.set_facecolor('#f6f6f6')

    plt.show()


def transactions_distribution(trans):
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#f6f5f5')
    ax = sns.histplot(data=trans, x='price', bins=50, stat="percent")
    ax.set_xlabel('Distribution of the price')
    for loc in ['bottom', 'left']:
        ax.spines[loc].set_visible(True)
        ax.spines[loc].set_linewidth(2)
        ax.spines[loc].set_color('black')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()
