import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np 
from scipy.stats import kurtosis, skew 
from scipy import stats

import matplotlib.pyplot as plt 
import seaborn as sns 

import plotly.offline as plty
from plotly import tools
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot, plot 
import plotly.graph_objs as go 

plt.style.use('fivethirtyeight')
init_notebook_mode(connected=True)
sns.set_style("whitegrid")
sns.set_context("paper")

from datetime import datetime

def DataDesc(df):
    """
    Создает сводную таблицу для pandas DataFrame, предоставляющую информацию о качестве и составе данных.

    Функция генерирует таблицу, которая включает:
    1. Тип данных каждого столбца.
    2. Количество отсутствующих значений в каждом столбце.
    3. Количество уникальных значений в каждом столбце.
    4. Первое по популярности значение в каждом столбце.
    5. Второе по популярности значение в каждом столбце.
    6. Третье по популярности значение в каждом столбце.

    Параметры:
        df (pandas.DataFrame): Датафрейм, для которого необходимо сгенерировать сводку.

    Возвращает:
        pandas.DataFrame: Таблица с сводной информацией по каждому столбцу датафрейма.
    """
    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values
    
    return summary

def Null_Count(df):
    """
    Подсчитывает количество и процентное соотношение отсутствующих значений (null) для каждой колонки в DataFrame.

    Эта функция анализирует DataFrame на наличие null значений в каждой колонке, сортирует их по убыванию и
    возвращает DataFrame, содержащий названия колонок, количество null значений в каждой из них,
    а также процентное соотношение этих null значений от общего числа записей в колонке.

    Параметры:
        df (pandas.DataFrame): Датафрейм, в котором необходимо подсчитать отсутствующие значения.

    Возвращает:
        pandas.DataFrame: Таблица с колонками 'Column', 'Null_Count' и 'Null_Percent', включающая только те колонки,
                          где количество отсутствующих значений больше нуля. Колонки отсортированы по убыванию 
                          количества отсутствующих значений.
    """
    df_null = df.isnull().sum().sort_values(ascending = False).rename('Null').reset_index()

    null_count = df_null['Null']
    null_percent = (null_count * 100) / (df.shape[0])

    df_null = pd.concat([df_null['index'],null_count,null_percent], axis=1, keys=['Column','Null_Count','Null_Percent'])

    return df_null[df_null['Null_Count'] != 0]

def fill_na(df): 
    '''
    Функция для заполнения нанов

    Примечание : булевые значения заполняются True
    '''
    df['totals_pageviews'].fillna(1, inplace=True)
    df['totals_newVisits'].fillna(0, inplace=True)
    df['totals_bounces'].fillna(0, inplace=True) 
    df["totals_transactionRevenue"].fillna(0.0, inplace=True)
    
    df['totals_pageviews'] = df['totals_pageviews'].astype(int)
    df['totals_newVisits'] = df['totals_newVisits'].astype(int)
    df['totals_bounces'] = df['totals_bounces'].astype(int)
    df["totals_transactionRevenue"] = df["totals_transactionRevenue"].astype(float)
    
    
    df['trafficSource_isTrueDirect'].fillna(False, inplace=True) 
    df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    df[df['geoNetwork_city'] == "(not set)"]['geoNetwork_city'] = np.nan
    df['geoNetwork_city'].fillna("NaN", inplace=True)
    
    return df

def date_process(df):
    """
    Заполняет отсутствующие значения (NaN) в определенных колонках DataFrame заданными значениями.

    Функция специализированно обрабатывает колонки, связанные с веб-аналитикой и источниками трафика,
    присваивая логически обоснованные значения в случае их отсутствия*:
    - 'totals_pageviews': пропущенные значения заменяются на 1 (минимально возможное количество просмотров).
    - 'totals_newVisits', 'totals_bounces': пропущенные значения заменяются на 0.
    - 'totals_transactionRevenue': пропущенные значения заменяются на 0.0.
    - 'trafficSource_isTrueDirect': пропущенные значения заменяются на False.
    - 'trafficSource_adwordsClickInfo.isVideoAd': пропущенные значения заменяются на True.

    *Note : nan в колонках с типом bool заменяются на противоположные существующим, например в колонке
    trafficSource_adwordsClickInfo.isVideoAd только два значения [np.nan, False], поэтому значения заменяются на True.

    Колонка 'geoNetwork_city' обрабатывается особым образом:
    - Значения "(not set)" заменяются на NaN, после чего все NaN заменяются на строку "NaN".

    Параметры:
        df (pandas.DataFrame): Датафрейм, в котором нужно заполнить отсутствующие значения.

    Возвращает:
        pandas.DataFrame: Датафрейм с заполненными значениями.
    """

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d") 
    df["weekday"] = df['date'].dt.weekday 
    df["day"] = df['date'].dt.day
    df["month"] = df['date'].dt.month
    df["year"] = df['date'].dt.year
    df['visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
    
    return df

class WebColor:
    PURPLE = '#800080'  
    CYAN = '#00FFFF'    
    DARKCYAN = '#008B8B'  
    BLUE = '#0000FF'    
    GREEN = '#008000'   
    YELLOW = '#FFFF00'  
    RED = '#FF0000'     

def PieChart(df, df_column, title, limit=15):
    """
    Строит две круговые диаграммы (пайчарты) для заданной колонки DataFrame: одну по количеству значений,
    а другую по суммарным доходам, связанным с каждым значением этой колонки.

    Параметры:
        df (pandas.DataFrame): Датафрейм, содержащий данные для анализа.
        df_column (str): Название колонки в DataFrame, для которой будут построены диаграммы.
        title (str): Заголовок для графиков.
        limit (int, optional): Максимальное количество категорий для отображения на диаграмме количества. По умолчанию 15.

    Описание:
        Функция сначала считает количество вхождений для каждого уникального значения в указанной колонке и строит
        по ним круговую диаграмму. Затем считает суммарный доход по каждой категории и строит вторую круговую диаграмму
        для наибольших по доходу категорий. Диаграммы располагаются горизонтально.

    Визуализация:
        - Первая диаграмма (слева) показывает процентное соотношение количества вхождений для топ-категорий.
        - Вторая диаграмма (справа) показывает процентное соотношение суммарных доходов для категорий с наибольшими доходами.

    Примечания:
        - Цвета для диаграмм заданы списками `colors_visits` и `colors_revenue`.
        - Настройка дизайна и макета графика выполняется через `layout`.
        - Функция использует библиотеку Plotly для визуализации.
    """

    count_trace = df[df_column].value_counts()[:limit].to_frame().reset_index()
    count_trace.columns = ['category', 'count'] 

    rev_trace = df.groupby(df_column)["totals_transactionRevenue"].sum().nlargest(10).reset_index()
    rev_trace.columns = [df_column, 'totals_transactionRevenue_log']

    colors_visits = [WebColor.PURPLE, WebColor.CYAN, WebColor.DARKCYAN, WebColor.BLUE,
                     WebColor.GREEN, WebColor.YELLOW, WebColor.RED, WebColor.PURPLE,
                     WebColor.CYAN, WebColor.DARKCYAN]
    colors_revenue = [WebColor.BLUE, WebColor.GREEN, WebColor.YELLOW, WebColor.RED,
                      WebColor.PURPLE, WebColor.CYAN, WebColor.DARKCYAN, WebColor.BLUE,
                      WebColor.GREEN, WebColor.YELLOW]

    trace1 = go.Pie(labels=count_trace['category'], values=count_trace['count'], name="% Accesses", hole=0.5,
                    hoverinfo="label+percent+name", showlegend=True, domain={'x': [0, .48]},
                    marker=dict(colors=colors_visits))

    trace2 = go.Pie(labels=rev_trace[df_column], values=rev_trace['totals_transactionRevenue_log'], name="% Revenue", hole=0.5,
                    hoverinfo="label+percent+name", showlegend=False, domain={'x': [.52, 1]},
                    marker=dict(colors=colors_revenue))

    layout = dict(title=title, height=450, font=dict(size=15),
                  annotations=[
                      dict(x=.25, y=.5, text='Visits', showarrow=False, font=dict(size=20)),
                      dict(x=.80, y=.5, text='Revenue', showarrow=False, font=dict(size=20))
                  ])

    fig = dict(data=[trace1, trace2], layout=layout)
    iplot(fig)

def add_aggregated_features(df, aggs):
    """
    Агрегирует данные в DataFrame по 'fullVisitorId', применяя заданные агрегатные функции,
    и добавляет полученные агрегированные значения обратно к исходному DataFrame.

    Параметры:
        df (pandas.DataFrame): Исходный DataFrame, в котором происходит агрегация.
        aggs (dict): Словарь, где ключи — это названия колонок для агрегации, а значения —
                     список функций агрегирования, которые нужно применить к этим колонкам.
                     Пример: {'totals_hits': ['sum', 'max'], 'totals_pageviews': ['mean', 'sum']}

    Описание:
        - Функция группирует данные по 'fullVisitorId', применяя заданные агрегатные функции.
        - Имена новых агрегированных колонок формируются путем конкатенации имени колонки и названия
          функции агрегирования (например, 'totals_hits_sum', 'totals_hits_max').
        - Результат агрегирования объединяется с исходным DataFrame по 'fullVisitorId'.

    Возвращает:
        pandas.DataFrame: Расширенный DataFrame, который включает исходные данные и добавленные агрегированные колонки.

    Примеры вывода:
        - Перед выводом функция отображает размер исходного DataFrame.
        - Отображает первые строки агрегированного DataFrame.
        - После объединения показывает размер измененного DataFrame.
    """
    print("Original DataFrame shape:", df.shape)
    grouped = df.groupby('fullVisitorId').agg(aggs)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    print("Aggregated DataFrame sample:", grouped.head())
    grouped.reset_index(inplace=True)
    df = df.merge(grouped, on='fullVisitorId', how='left')
    
    print("Modified DataFrame shape after merge:", df.shape) 
    
    return df