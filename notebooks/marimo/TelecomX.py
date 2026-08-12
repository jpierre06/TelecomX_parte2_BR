import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📌 Importação de bibliotecas
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return np, pd


@app.cell
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt

    return plt, sns


@app.cell
def _():
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    return


@app.cell
def _():
    from scipy.stats import f_oneway
    from scipy.stats import levene
    import pingouin as pg

    return


@app.cell
def _():
    from scipy.stats import variation, skew, kurtosis, shapiro, entropy

    return entropy, kurtosis, shapiro, skew, variation


@app.cell
def _():
    from optbinning import OptimalBinning
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_validate, KFold
    from sklearn.model_selection import StratifiedKFold
    from imblearn.pipeline import Pipeline as imbpipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import NearMiss
    from yellowbrick.model_selection import FeatureImportances
    from sklearn.dummy import DummyClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import RocCurveDisplay
    from sklearn.metrics import PrecisionRecallDisplay
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import classification_report
    from yellowbrick.classifier import ClassificationReport
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve

    return GridSearchCV, RandomForestClassifier, train_test_split


@app.cell
def _():
    import warnings

    return (warnings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importação de pacotes locais
    """)
    return


@app.cell
def _():
    import utils.local_tools as lt
    import telecomx.machine_learning as tml

    return lt, tml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configurações do ambiente
    """)
    return


@app.cell
def _(pd):
    pd.set_option('display.max_columns', None)
    return


@app.cell
def _(warnings):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Constantes
    """)
    return


@app.cell
def _():
    NUM_SEMENTE_ALEATORIA = 42
    TAMANHO_TESTE = 0.3
    MAXIMO_ITERACAO = 1000
    LIMITE_CATEGORICAS = 12
    return LIMITE_CATEGORICAS, NUM_SEMENTE_ALEATORIA, TAMANHO_TESTE


@app.cell
def _():
    LIST_SCORING = ['accuracy','recall', 'precision', 'f1']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📌 Extração de dados
    """)
    return


@app.cell
def _(pd):
    df_dados = pd.read_csv('./dados/dados_tratados.csv')
    return (df_dados,)


@app.cell
def _(df_dados):
    df_dados.info()
    return


@app.cell
def _(df_dados):
    df_dados.nunique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verificar dados duplicados
    """)
    return


@app.cell
def _(df_dados):
    df_dados.duplicated().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📌 Tratamento de dados
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tratando dados de gênero
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Criando cópia da base de dadaos
    """)
    return


@app.cell
def _(df_dados):
    df_clean = df_dados.copy()
    return (df_clean,)


@app.cell
def _(df_clean):
    df_clean.customer_gender = df_clean.customer_gender.map({'Female': 1, 'Male': 0})
    return


@app.cell
def _(df_clean):
    df_clean.drop(columns=['customerID'], inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📌 Análise Exploratória de Dados (EDA)
    """)
    return


@app.cell
def _(df_clean, plt, sns):
    sns.countplot(data=df_clean, x='Churn')
    plt.title('Distribuição do Alvo (Churn)')
    plt.xticks([0, 1], ['Permaneceu', 'Evasão'])
    plt.ylabel('Número de Clientes')

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visão geral do alvo (Churn)
    """)
    return


@app.cell
def _(df_clean):
    # Porcentagem
    df_clean['Churn'].value_counts(normalize=True) * 100
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Como balancear os dados da variável alvo?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Análise univariada: variáveis categóricas
    """)
    return


@app.cell
def _(df_clean, plt, sns):
    sns.countplot(x = df_clean.customer_gender.map({1: 'Female', 0: 'Male'}), hue = df_clean.Churn.map({1: 'Yes', 0: 'No'}))
    plt.title('Churn por Gênero')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Análise univariada: variáveis numéricas
    """)
    return


@app.cell
def _(df_clean, lt):
    bins = lt.get_bins_rule_sturges(df_clean.account_Charges_Monthly)
    bins
    return (bins,)


@app.cell
def _(bins, df_clean, plt, sns):
    sns.histplot(data=df_clean, x='account_Charges_Monthly', hue='Churn', kde=True, binwidth=bins)
    plt.title('Distribuição das Cobranças Mensais por Churn')
    plt.show()
    return


@app.cell
def _(df_clean, plt, sns):
    sns.boxplot(data=df_clean, x='Churn', y='account_Charges_Monthly')
    plt.title('Boxplot - Cobrança Mensal vs. Churn')
    plt.show()
    return


@app.cell
def _(df_clean, plt, sns):
    sns.boxplot(data=df_clean, x='Churn', y='customer_tenure')
    plt.title('Boxplot - Tempo de Contrato vs. Churn')
    plt.show()
    return


@app.cell
def _(df_clean, plt, sns):
    sns.boxplot(data=df_clean, x='Churn', y='account_Charges_Total')
    plt.title('Boxplot - Gasto Total vs. Churn')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlações
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Somente com variáveis numéricas:
    """)
    return


@app.cell
def _(df_clean, np):
    colunas_analise = df_clean.select_dtypes(include=[np.number]).columns
    return (colunas_analise,)


@app.cell
def _(colunas_analise, df_clean, tml):
    tml.graf_correlacao_variaveis(df_clean, colunas_analise, tam_figura=(14,14), tam_fonte=10).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlação do tipo de serviço
    """)
    return


@app.cell
def _():
    colunas_analise_1 = ['phone_PhoneService', 'phone_MultipleLines', 'internet_InternetService', 'only_PhoneService', 'only_InternetService', 'both_Phone_InternetService']
    return (colunas_analise_1,)


@app.cell
def _(colunas_analise_1, df_clean):
    df_clean[colunas_analise_1].head()
    return


@app.cell
def _(colunas_analise_1, df_clean, tml):
    tml.graf_correlacao_variaveis(df_clean, colunas_analise_1).show()
    return


@app.cell
def _(colunas_analise_1, df_clean, lt):
    lt.get_variance_inflation_factor(df_clean, colunas_analise_1)
    return


@app.cell
def _(colunas_analise_1, df_clean, lt):
    [colunas_analise_1.remove(x) for x in ['only_PhoneService', 'only_InternetService', 'both_Phone_InternetService']]
    lt.get_variance_inflation_factor(df_clean, colunas_analise_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlação de valores de serviço
    """)
    return


@app.cell
def _():
    colunas_analise_2 = ['account_Charges_Monthly', 'account_Charges_Total', 'account_Charges_Daily', 'customer_tenure']
    return (colunas_analise_2,)


@app.cell
def _(colunas_analise_2, df_clean):
    df_clean[colunas_analise_2].head()
    return


@app.cell
def _(colunas_analise_2, df_clean, tml):
    tml.graf_correlacao_variaveis(df_clean, colunas_analise_2).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verificando fator de inflação de variância
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✅ 2. Limiares comuns para suspeita de multicolinearidade

    | Indicador                       | Limite comum    | Interpretação                            |
    | ------------------------------- | --------------- | ---------------------------------------- |
    | Correlação entre variáveis      | > 0.8 ou < -0.8 | Alta correlação direta ou inversa        |
    | VIF (Variance Inflation Factor) | > 5 (ou 10)     | Indica potencial multicolinearidade      |
    | Tolerância (1 / VIF)            | < 0.2           | Problema potencial de multicolinearidade |
    """)
    return


@app.cell
def _(colunas_analise_2, df_clean, lt):
    lt.get_variance_inflation_factor(df_clean, colunas_analise_2)
    return


@app.cell
def _(colunas_analise_2, df_clean, lt):
    [colunas_analise_2.remove(x) for x in ['account_Charges_Daily']]
    lt.get_variance_inflation_factor(df_clean, colunas_analise_2)
    return


@app.cell
def _(colunas_analise_2, df_clean, lt):
    [colunas_analise_2.remove(x) for x in ['account_Charges_Total']]
    lt.get_variance_inflation_factor(df_clean, colunas_analise_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlação de características do cliente e da família
    """)
    return


@app.cell
def _():
    colunas_analise_3 = ['customer_gender', 'customer_SeniorCitizen', 'customer_Partner', 'customer_Dependents']
    return (colunas_analise_3,)


@app.cell
def _(colunas_analise_3, df_clean):
    df_clean[colunas_analise_3].head()
    return


@app.cell
def _(colunas_analise_3, df_clean, tml):
    tml.graf_correlacao_variaveis(df_clean, colunas_analise_3).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verificando fator de inflação de variância
    """)
    return


@app.cell
def _(colunas_analise_3, df_clean, lt):
    lt.get_variance_inflation_factor(df_clean, colunas_analise_3, False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlação dos serviços adicionais de internet
    """)
    return


@app.cell
def _():
    colunas_analise_4 = ['internet_OnlineSecurity', 'internet_OnlineBackup', 'internet_DeviceProtection', 'internet_TechSupport', 'internet_StreamingTV', 'internet_StreamingMovies', 'additional_InternetService']
    return (colunas_analise_4,)


@app.cell
def _(colunas_analise_4, df_clean):
    df_clean[colunas_analise_4].head()
    return


@app.cell
def _(colunas_analise_4, df_clean, tml):
    tml.graf_correlacao_variaveis(df_clean, colunas_analise_4).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verificando fator de inflação de variância
    """)
    return


@app.cell
def _(colunas_analise_4, df_clean, lt):
    lt.get_variance_inflation_factor(df_clean, colunas_analise_4)
    return


@app.cell
def _(colunas_analise_4, df_clean, lt):
    [colunas_analise_4.remove(x) for x in ['internet_OnlineSecurity', 'internet_OnlineBackup', 'internet_DeviceProtection', 'internet_TechSupport', 'internet_StreamingTV', 'internet_StreamingMovies']]
    lt.get_variance_inflation_factor(df_clean, colunas_analise_4)
    return


@app.cell
def _():
    colunas_analise_5 = ['internet_OnlineSecurity', 'internet_OnlineBackup', 'internet_DeviceProtection', 'internet_TechSupport', 'internet_StreamingTV', 'internet_StreamingMovies', 'additional_InternetService']
    return (colunas_analise_5,)


@app.cell
def _(colunas_analise_5, df_clean, lt):
    [colunas_analise_5.remove(x) for x in ['additional_InternetService']]
    lt.get_variance_inflation_factor(df_clean, colunas_analise_5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📌 Estatística Descritiva e Inferencial
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estatística Descritiva
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Medidas de tendência central e dispersão
    """)
    return


@app.cell
def _(df_dados):
    df_dados.groupby('Churn')[['account_Charges_Monthly', 'customer_tenure']].describe().T
    return


@app.cell
def _(df_dados, lt):
    lt.describe_full_df_segmented(df_dados, 'account_Charges_Monthly', 'Churn', extend_metrics=True)
    return


@app.cell
def _(df_dados, lt):
    lt.describe_full_df_segmented(df_dados, 'customer_tenure', 'Churn')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coeficiente de variação, assimetria, normalidade e variabilidade
    """)
    return


@app.cell
def _(df_clean, entropy, kurtosis, shapiro, skew, variation):
    colunas_analise_6 = ['account_Charges_Monthly', 'account_Charges_Total', 'customer_tenure']
    print(f'Observações: {len(df_clean)}')
    for _c in colunas_analise_6:
        cv = variation(df_clean[_c])
        sk = skew(df_clean[_c])
        kt = kurtosis(df_clean[_c])
        sp_t, p = shapiro(df_clean[_c])
        et = entropy(df_clean[_c])
        print(f'\n{_c}:')
        print(f'Coeficiente de Variação: {cv:.2f}')
        print(f'Assimetria (Skew): {sk:.2f}')
        print(f'Kurtosis (Skew): {kt:.2f}')
        print(f'Shapiro Wilk - Estatística: {sp_t:.2f}. P-value: {p:.2f}.')
        print(f'Entropia: {et:.2f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Variável                      | Coef. Variação (CV)       | Assimetria (Skew)                              | Curtose                                       | Normalidade (Shapiro-Wilk) | Entropia                | Observações e Impacto no ML                                                                                                                     | Necessidade de Tratamento                                                                           |
    | ----------------------------- | ------------------------- | ---------------------------------------------- | --------------------------------------------- | -------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
    | **account\_Charges\_Monthly** | 0.46 (baixa dispersão)    | -0.22 (quase simétrica, leve cauda à esquerda) | -1.26 (distribuição mais achatada que normal) | p < 0.05 → não normal      | 8.74 (alta diversidade) | Boa estabilidade, pouca variabilidade relativa; ausência de normalidade não é problema grave para modelos baseados em árvore.                   | Escalonar apenas se usar modelos sensíveis a escala (ex.: regressão logística, SVM, redes neurais). |
    | **account\_Charges\_Total**   | 0.99 (alta dispersão)     | 0.96 (assimetria positiva moderada)            | -0.23 (levemente achatada)                    | p < 0.05 → não normal      | 8.37 (alta diversidade) | Alta dispersão e assimetria indicam que poucos clientes concentram valores altos; pode ser preditiva de churn se associada a tempo de contrato. | Avaliar log-transform ou binning se usar modelos lineares; árvores lidam bem.                       |
    | **customer\_tenure**          | 0.76 (moderada dispersão) | 0.24 (quase simétrica)                         | -1.39 (muito achatada)                        | p < 0.05 → não normal      | 8.53 (alta diversidade) | Distribuição relativamente equilibrada, mas achatada; mostra clientes distribuídos em várias faixas de tempo.                                   | Pode manter contínua; binning pode ser útil se relação com churn não for linear.                    |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    📌 Principais pontos para churn

        Nenhuma variável é normal — mas para modelos como XGBoost, LightGBM ou Random Forest, isso não é um problema.

        account_Charges_Total é a mais assimétrica → pode precisar de transformação apenas se usar modelos sensíveis à escala ou regressões.

        customer_tenure já foi binada antes por você — manter bins pode melhorar interpretação, mas perde um pouco da granularidade.

        CV mostra que account_Charges_Monthly é mais estável, enquanto account_Charges_Total é bem mais volátil (potencialmente mais informativo).

        Entropia alta em todas indica boa variabilidade de valores (bom para ML).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Crição de binned otimizados para variáveis numéricas
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Deletar colunas bins criadas anteriormente e criar novas bins otimizadas
    """)
    return


@app.cell
def _(df_clean):
    df_clean.drop(columns=['customer_tenure_bins', 'account_Charges_Monthly_bins', 'account_Charges_Total_bins'], inplace=True)
    return


@app.cell
def _(df_clean, lt):
    df_clean['tenure_bin'] = lt.optimal_bins(df_clean, 'customer_tenure', 'Churn')
    df_clean['charges_monthly_bin'] = lt.optimal_bins(df_clean, 'account_Charges_Monthly', 'Churn')
    df_clean['charges_total_bin'] = lt.optimal_bins(df_clean, 'account_Charges_Total', 'Churn')
    return


@app.cell
def _(df_clean):
    df_clean.head()
    return


@app.cell
def _(df_clean, lt):
    df_clean_1 = lt.adjust_column_names(df_clean)
    return (df_clean_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estatística Inferencial
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Teste Qui-Quadrado (duas variáveis categóricas)
    """)
    return


@app.cell
def _(LIMITE_CATEGORICAS, df_clean_1, lt):
    variaveis_categoricas, dominios = lt.get_columns_categorical(df_clean_1, LIMITE_CATEGORICAS)
    return dominios, variaveis_categoricas


@app.cell
def _(dominios, variaveis_categoricas):
    list(zip(variaveis_categoricas, dominios))
    return


@app.cell
def _(df_clean_1, lt, variaveis_categoricas):
    lt.get_chi_square(df_clean_1, variaveis_categoricas[0], variaveis_categoricas[1:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Teste T de médias (variável numérica vs binária)
    """)
    return


@app.cell
def _(LIMITE_CATEGORICAS, df_clean_1, lt):
    variaveis_numericas = lt.get_columns_numeric(df_clean_1, LIMITE_CATEGORICAS)
    variaveis_numericas
    return (variaveis_numericas,)


@app.cell
def _(df_clean_1, lt, variaveis_numericas):
    lt.get_ttest_ind(df_clean_1, 'Churn', variaveis_numericas)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verificação de multicolinearidade com VIF
    """)
    return


@app.cell
def _(df_clean_1, lt, variaveis_numericas):
    lt.get_variance_inflation_factor(df_clean_1, variaveis_numericas)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✅ Conclusão

        Sim, com um VIF > 9, a variável account_Charges_Total apresenta forte multicolinearidade com outras variáveis do modelo.

        Causa provável:
        account_Charges_Total é provavelmente o produto de account_Charges_Monthly * customer_tenure, ou algo muito próximo disso. Portanto, ela está matematicamente associada às outras duas, o que gera colinearidade.

    | Variável                  | VIF      | Interpretação                       |
    | ------------------------- | -------- | ----------------------------------- |
    | `const`                   | 14.90    | Desconsiderar, é apenas a constante |
    | `account_Charges_Total`   | **9.51** | 🚨 Alta multicolinearidade          |
    | `customer_tenure`         | 5.84     | Moderada multicolinearidade         |
    | `account_Charges_Monthly` | 3.22     | Aceitável                           |
    """)
    return


@app.cell
def _(df_clean_1, lt, variaveis_numericas):
    temp = variaveis_numericas.copy()
    [temp.remove(x) for x in ['account_Charges_Total', 'account_Charges_Daily']]
    lt.get_variance_inflation_factor(df_clean_1, temp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fator de inflação de variância (VIF)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Colunas binárias
    """)
    return


@app.cell
def _(df_clean_1, lt):
    variaveis_binarias = lt.get_columns_binary(df_clean_1)
    variaveis_binarias[0]
    return (variaveis_binarias,)


@app.cell
def _(variaveis_binarias):
    # Colunas binárias sem Churn
    variaveis_binarias[1:]
    return


@app.cell
def _(variaveis_binarias):
    temp_1 = variaveis_binarias.copy()
    [temp_1.remove(x) for x in ['Churn', 'customer_gender', 'only_PhoneService', 'only_InternetService', 'both_Phone_InternetService', 'internet_OnlineSecurity', 'internet_OnlineBackup', 'internet_DeviceProtection', 'internet_TechSupport', 'internet_StreamingTV', 'internet_StreamingMovies']]
    return (temp_1,)


@app.cell
def _(df_clean_1, lt, temp_1):
    lt.get_variance_inflation_factor(df_clean_1, temp_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Análise de multicolinearidade de variáveis derivadas
    """)
    return


@app.cell
def _(LIMITE_CATEGORICAS, df_clean_1, lt):
    variaveis_categoricas_1, dominios_1 = lt.get_columns_categorical(df_clean_1, LIMITE_CATEGORICAS)
    return (variaveis_categoricas_1,)


@app.cell
def _(variaveis_binarias, variaveis_categoricas_1):
    variaveis_multi_categoricas = list(set(variaveis_categoricas_1).difference(variaveis_binarias))
    variaveis_multi_categoricas
    return (variaveis_multi_categoricas,)


@app.cell
def _(variaveis_multi_categoricas):
    lista_otimizada = variaveis_multi_categoricas.copy()
    lista_otimizada.extend(['customer_tenure', 'account_Charges_Monthly', 'account_Contract_Monthly', 'internet_InternetService'])
    return (lista_otimizada,)


@app.cell
def _(variaveis_multi_categoricas):
    variaveis_multi_categoricas
    return


@app.cell
def _(lista_otimizada):
    lista_otimizada
    return


@app.cell
def _(df_clean_1, lista_otimizada, pd, variaveis_multi_categoricas):
    temp_ohe = pd.get_dummies(df_clean_1[lista_otimizada], columns=variaveis_multi_categoricas, drop_first=False, dtype=int)
    return (temp_ohe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlação dos meses de contratos
    """)
    return


@app.cell
def _(lt, temp_ohe):
    colunas_analise_7 = lt.filter_startswith(temp_ohe.columns, 'tenure_bin_')
    return (colunas_analise_7,)


@app.cell
def _(colunas_analise_7, temp_ohe):
    temp_ohe[colunas_analise_7].head()
    return


@app.cell
def _(colunas_analise_7, temp_ohe, tml):
    tml.graf_correlacao_variaveis(temp_ohe, colunas_analise_7).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verificando fator de inflação de variância
    """)
    return


@app.cell
def _(colunas_analise_7, lt, temp_ohe):
    lt.get_variance_inflation_factor(temp_ohe, colunas_analise_7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlação dos custos mensais
    """)
    return


@app.cell
def _(lt, temp_ohe):
    colunas_analise_8 = lt.filter_startswith(temp_ohe.columns, 'charges_monthly_bin_')
    return (colunas_analise_8,)


@app.cell
def _(colunas_analise_8, temp_ohe):
    temp_ohe[colunas_analise_8].head()
    return


@app.cell
def _(colunas_analise_8, temp_ohe, tml):
    tml.graf_correlacao_variaveis(temp_ohe, colunas_analise_8).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verificando fator de inflação de variância
    """)
    return


@app.cell
def _(colunas_analise_8, lt, temp_ohe):
    lt.get_variance_inflation_factor(temp_ohe, colunas_analise_8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlação dos custos totais
    """)
    return


@app.cell
def _(lt, temp_ohe):
    colunas_analise_9 = lt.filter_startswith(temp_ohe.columns, 'charges_total_bin_')
    return (colunas_analise_9,)


@app.cell
def _(colunas_analise_9, temp_ohe):
    temp_ohe[colunas_analise_9].head()
    return


@app.cell
def _(colunas_analise_9, temp_ohe, tml):
    tml.graf_correlacao_variaveis(temp_ohe, colunas_analise_9).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verificando fator de inflação de variância
    """)
    return


@app.cell
def _(colunas_analise_9, lt, temp_ohe):
    lt.get_variance_inflation_factor(temp_ohe, colunas_analise_9)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlação dos Serviços de Internet
    """)
    return


@app.cell
def _():
    colunas_analise_10 = ['internet_InternetService', 'internet_Service_Description_DSL', 'internet_Service_Description_Fiber optic', 'internet_Service_Description_No']
    return (colunas_analise_10,)


@app.cell
def _(colunas_analise_10, temp_ohe):
    temp_ohe[colunas_analise_10].head()
    return


@app.cell
def _(colunas_analise_10, temp_ohe):
    temp_ohe[colunas_analise_10].query('`internet_Service_Description_DSL` == 0 and `internet_Service_Description_Fiber optic` == 0').head()
    return


@app.cell
def _(colunas_analise_10, temp_ohe, tml):
    tml.graf_correlacao_variaveis(temp_ohe, colunas_analise_10).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verificando fator de inflação de variância
    """)
    return


@app.cell
def _(colunas_analise_10, lt, temp_ohe):
    lt.get_variance_inflation_factor(temp_ohe, colunas_analise_10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Seleção de fatures através da correlação
    """)
    return


@app.cell
def _(colunas_analise_10, lt, temp_ohe):
    [colunas_analise_10.remove(x) for x in ['internet_Service_Description_No', 'internet_InternetService']]
    lt.get_variance_inflation_factor(temp_ohe, colunas_analise_10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlação dos Tipos de Contratos
    """)
    return


@app.cell
def _():
    colunas_analise_11 = ['account_Contract_Monthly', 'account_Contract_Month-to-month', 'account_Contract_One year', 'account_Contract_Two year']
    return (colunas_analise_11,)


@app.cell
def _(colunas_analise_11, temp_ohe):
    temp_ohe[colunas_analise_11].head()
    return


@app.cell
def _(colunas_analise_11, temp_ohe, tml):
    tml.graf_correlacao_variaveis(temp_ohe, colunas_analise_11).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verificando fator de inflação de variância
    """)
    return


@app.cell
def _(colunas_analise_11, lt, temp_ohe):
    lt.get_variance_inflation_factor(temp_ohe, colunas_analise_11)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ANOVA - Análise de Variáveis independentes (categóricas x numéricas)
    """)
    return


@app.cell
def _(LIMITE_CATEGORICAS, df_clean_1, lt):
    variaveis_categoricas_2, dominios_2 = lt.get_columns_categorical(df_clean_1, LIMITE_CATEGORICAS)
    return (variaveis_categoricas_2,)


@app.cell
def _(variaveis_binarias, variaveis_categoricas_2):
    variaveis_multi_categoricas_1 = list(set(variaveis_categoricas_2).difference(variaveis_binarias))
    variaveis_multi_categoricas_1
    return (variaveis_multi_categoricas_1,)


@app.cell
def _(df_clean_1, lt, variaveis_multi_categoricas_1):
    lt.print_tabulate_df(lt.get_anova_levene_test(df_clean_1, 'anova', ['account_Charges_Total'], variaveis_multi_categoricas_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Verificar Homogeneidade de variâncias (homocedasticidade)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Teste de Levene
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use o teste de Levene:

    Se p < 0.05, as variâncias são diferentes → nesse caso, prefira o teste de Welch ANOVA (statsmodels ou pingouin), que é robusto a isso.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Etapa                        | Resultado | Ação recomendada                                   |
    | ---------------------------- | --------- | -------------------------------------------------- |
    | ANOVA (`f_oneway`)           | p < 0.05  | Há diferença entre grupos                          |
    | Teste de Levene (variâncias) | p < 0.05  | Variâncias diferentes → ANOVA tradicional inválida |
    | Alternativa                  | —         | Use **Welch ANOVA** e **Games-Howell**             |
    """)
    return


@app.cell
def _(df_clean_1, lt, variaveis_multi_categoricas_1):
    lt.print_tabulate_df(lt.get_anova_levene_test(df_clean_1, 'levene', ['account_Charges_Total'], variaveis_multi_categoricas_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Welch ANOVA e Games-Howell Post-hoc
    """)
    return


@app.cell
def _(df_clean_1, lt, variaveis_multi_categoricas_1):
    lt.get_welch_anova_test(df_clean_1, 'Churn', variaveis_multi_categoricas_1)
    return


@app.cell
def _(df_clean_1, lt, variaveis_multi_categoricas_1):
    lt.get_gameshowell_test(df_clean_1, 'Churn', variaveis_multi_categoricas_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✅ O que os testes indicam

        * Welch ANOVA

            Todos os testes mostraram valores altamente significativos (p < 0.001), indicando que há diferenças estatisticamente significativas nas médias de Churn entre os grupos de cada variável categórica analisada.

        * Games-Howell Post Hoc

            Mostrou diferenças significativas entre diversos pares de grupos com tamanhos de efeito (hedges) de moderado a alto, o que reforça a relevância dessas variáveis na separação de grupos com diferentes comportamentos de Churn.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📌 Tratamento de dados
    """)
    return


@app.cell
def _(df_dados, tml):
    df_ohe = tml.df_final_modelo_v1(df_dados)
    return (df_ohe,)


@app.cell
def _(df_ohe, tml):
    tml.graf_correlacao_variaveis(df_ohe, df_ohe.columns, tam_figura=(14,14), tam_fonte=9).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📌 Seleção e validação dos modelos de treinamento
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tabela de verificação de padronização dos dados
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Modelo                 | Precisa padronizar? | Tipo de padronização (se necessário)   |
    | ---------------------- | ------------------- | -------------------------------------- |
    | DecisionTreeClassifier | ❌ Não               | —                                      |
    | LogisticRegression     | ✅ Sim               | `StandardScaler` (média 0, desvio 1)   |
    | RandomForest           | ❌ Não               | —                                      |
    | XGBoost                | ⚠️ Opcional         | Pode ajudar (StandardScaler ou MinMax) |
    | LightGBM               | ⚠️ Opcional         | Pode ajudar (StandardScaler ou MinMax) |
    | CatBoost               | ❌ Não               | —                                      |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conceitos sobre as métricas de um modelo
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Aqui está o quadro comparativo para churn (evasão de clientes), considerando que a classe positiva é “cliente vai sair”:

    | **Métrica**                | **O que mede**                                                                     | **Quando valor é alto**                                        | **Risco quando valor é baixo**                                   | **Custo de erro associado**                                                |
    | -------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
    | **Precisão (Precision)**   | Entre todos que o modelo previu como “vai sair”, qual porcentagem realmente saiu   | Você gasta retenção apenas em quem de fato sairia              | Gastar recursos em retenção de clientes que iam ficar (FP alto)  | **Custo financeiro** com campanhas desnecessárias (descontos, bônus, etc.) |
    | **Recall (Sensibilidade)** | Entre todos que realmente saíram, qual porcentagem o modelo previu como “vai sair” | Você identifica a maior parte dos clientes que iam sair        | Deixar escapar clientes que saem (FN alto)                       | **Perda de receita** e potencial perda de market share                     |
    | **F1-Score**               | Média harmônica de precisão e recall                                               | Equilíbrio entre acertar quem vai sair e evitar falsos alarmes | Ou alto custo de retenção inútil ou perda de clientes — ou ambos | **Equilíbrio financeiro e estratégico** — custo total menor                |
    | **FP (Falso Positivo)**    | Previu saída, mas o cliente ficaria                                                | —                                                              | Gastar para reter quem não ia sair                               | Desperdício de budget de retenção                                          |
    | **FN (Falso Negativo)**    | Previu permanência, mas o cliente saiu                                             | —                                                              | Não agir para reter quem realmente sairia                        | Perda direta de receita + possível impacto na reputação                    |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    📌 Resumo visual da prioridade

        Se retenção for muito cara → priorizar alta precisão.

        Se perder clientes for muito prejudicial → priorizar alto recall.

        Se quer balancear ambos → otimizar F1-Score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split base de dados em train e test
    """)
    return


@app.cell
def _(df_ohe):
    X = df_ohe.drop(columns=['Churn'])
    return (X,)


@app.cell
def _(df_ohe):
    y = df_ohe.Churn
    return (y,)


@app.cell
def _(NUM_SEMENTE_ALEATORIA, TAMANHO_TESTE, X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = NUM_SEMENTE_ALEATORIA, test_size=TAMANHO_TESTE, stratify=y)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Treinamento do Modelo - RandomForestClassifier
    """)
    return


@app.cell
def _(NUM_SEMENTE_ALEATORIA, RandomForestClassifier, X_train, y_train):
    model_rfc = RandomForestClassifier(max_depth=10, random_state=NUM_SEMENTE_ALEATORIA)
    model_rfc.fit(X_train, y_train)
    return (model_rfc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Otimizando os hiperparâmetros com o GridSearchCV
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Melhores parâmetros com base nas análises realizadas no notebook `TelecomX-RandomForest.ipynb`
    """)
    return


@app.cell
def _():
    best_params_ = {
        "bootstrap": [False],
        "class_weight": ["balanced"],
        "max_depth": [10],
        "min_samples_leaf": [2],
        "min_samples_split": [7],
        "n_estimators": [100]
    }
    return (best_params_,)


@app.cell
def _(GridSearchCV, best_params_, model_rfc):
    model_grid_rfc = GridSearchCV(
        estimator=model_rfc,
        param_grid=best_params_,
        scoring='recall',  # ou 'f1', 'roc_auc'
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    return (model_grid_rfc,)


@app.cell
def _(X_train, model_grid_rfc, y_train):
    model_grid_rfc.fit(X_train, y_train)
    return


@app.cell
def _(X_test, model_grid_rfc):
    y_pred_grid_rfc = model_grid_rfc.predict(X_test)
    return (y_pred_grid_rfc,)


@app.cell
def _(X_test, model_grid_rfc):
    y_pred_proba_grid_rfc = model_grid_rfc.predict_proba(X_test)
    return (y_pred_proba_grid_rfc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_pred_grid_rfc, y_pred_proba_grid_rfc, y_test):
    tml.df_classifier_metrics(y_test, y_pred_grid_rfc, y_pred_proba_grid_rfc, ['RandomForest - Grid'])
    return


@app.cell
def _(tml, y_pred_grid_rfc, y_pred_proba_grid_rfc, y_test):
    tml.avaliar_modelo(y_test, y_pred_grid_rfc, y_pred_proba_grid_rfc, True)
    return


@app.cell
def _(X_test, lt, pd, tml, y_pred_grid_rfc, y_test):
    list_df = []
    colunas_binarias = lt.get_columns_binary(X_test)
    for _c in colunas_binarias:
        list_df.append(tml.df_specific_confusion_matrix(X_test, y_test, y_pred_grid_rfc, _c))
    df_independent = pd.concat(list_df, axis=0)
    return (df_independent,)


@app.cell
def _(df_independent):
    df_independent
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Salvando modelo
    """)
    return


@app.cell
def _():
    import pickle

    return (pickle,)


@app.cell
def _(model_grid_rfc, pickle):
    with open('./modelos/model_grid_rfc.pkl', 'wb') as arquivo:
        pickle.dump(model_grid_rfc, arquivo)
    return


if __name__ == "__main__":
    app.run()
