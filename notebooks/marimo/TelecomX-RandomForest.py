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
    import pandas as pd

    return (pd,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return


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

    return (
        GridSearchCV,
        RandomForestClassifier,
        SMOTE,
        imbpipeline,
        train_test_split,
    )


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
    return NUM_SEMENTE_ALEATORIA, TAMANHO_TESTE


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


@app.cell
def _(df_dados, tml):
    df_ohe = tml.df_final_modelo_v1(df_dados)
    return (df_ohe,)


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


@app.cell
def _(model_rfc):
    model_rfc.get_params()
    return


@app.cell
def _(X_test, model_rfc):
    y_pred_rfc = model_rfc.predict(X_test)
    return (y_pred_rfc,)


@app.cell
def _(X_test, model_rfc):
    y_pred_proba_rfc = model_rfc.predict_proba(X_test)
    return (y_pred_proba_rfc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_pred_proba_rfc, y_pred_rfc, y_test):
    tml.df_classifier_metrics(y_test, y_pred_rfc, y_pred_proba_rfc, ['RandomForestClassifier'])
    return


@app.cell
def _(tml, y_pred_proba_rfc, y_pred_rfc, y_test):
    tml.avaliar_modelo(y_test, y_pred_rfc, y_pred_proba_rfc, print_graph=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Otimizando os hiperparâmetros com o GridSearchCV
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Teste 1
    """)
    return


@app.cell
def _():
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 7],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced']
    }
    return (param_grid,)


@app.cell
def _(GridSearchCV, model_rfc, param_grid):
    model_grid_rfc = GridSearchCV(
        estimator=model_rfc,
        param_grid=param_grid,
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
def _(model_grid_rfc):
    model_grid_rfc.best_params_
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
    tml.avaliar_modelo(y_test, y_pred_grid_rfc, y_pred_proba_grid_rfc, False)
    return


@app.cell
def _(X_test, lt, pd, tml, y_pred_grid_rfc, y_test):
    list_df = []
    colunas_binarias = lt.identify_columns_binary_values(X_test)
    for c in colunas_binarias:
        list_df.append(tml.df_specific_confusion_matrix(X_test, y_test, y_pred_grid_rfc, c))
    df_independent = pd.concat(list_df, axis=0)
    return (df_independent,)


@app.cell
def _(df_independent):
    df_independent.reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Teste 2
    """)
    return


@app.cell
def _(model_grid_rfc):
    model_grid_rfc.best_params_
    return


@app.cell
def _(param_grid):
    dict(sorted(param_grid.items()))
    return


@app.cell
def _():
    param_grid_1 = {'bootstrap': [False], 'class_weight': ['balanced'], 'max_depth': [15, 20], 'min_samples_leaf': [1, 3], 'min_samples_split': [2, 4, 9], 'n_estimators': [200, 300]}
    return (param_grid_1,)


@app.cell
def _(GridSearchCV, model_rfc, param_grid_1):
    model_grid_rfc2 = GridSearchCV(estimator=model_rfc, param_grid=param_grid_1, scoring='recall', cv=5, n_jobs=-1, verbose=2)  # ou 'f1', 'roc_auc'
    return (model_grid_rfc2,)


@app.cell
def _(X_train, model_grid_rfc2, y_train):
    model_grid_rfc2.fit(X_train, y_train)
    return


@app.cell
def _(model_grid_rfc2):
    model_grid_rfc2.best_params_
    return


@app.cell
def _(model_grid_rfc):
    model_grid_rfc.best_params_
    return


@app.cell
def _(X_test, model_grid_rfc2):
    y_pred_grid_rfc2 = model_grid_rfc2.predict(X_test)
    return (y_pred_grid_rfc2,)


@app.cell
def _(X_test, model_grid_rfc2):
    y_pred_proba_grid_rfc2 = model_grid_rfc2.predict_proba(X_test)
    return (y_pred_proba_grid_rfc2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_pred_grid_rfc2, y_pred_proba_grid_rfc2, y_test):
    tml.df_classifier_metrics(y_test, y_pred_grid_rfc2, y_pred_proba_grid_rfc2, ['RandomForest - Grid'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pipeline para validação Oversampling
    """)
    return


@app.cell
def _(NUM_SEMENTE_ALEATORIA, SMOTE, imbpipeline, model_grid_rfc):
    pipeline_over_rfc = imbpipeline([
        ('oversample', SMOTE(random_state=NUM_SEMENTE_ALEATORIA)),
        ('RandomForest', model_grid_rfc),
    ])
    return (pipeline_over_rfc,)


@app.cell
def _(X_train, pipeline_over_rfc, y_train):
    pipeline_over_rfc.fit(X_train, y_train)
    return


@app.cell
def _(X_test, pipeline_over_rfc):
    y_pred_over_rfc = pipeline_over_rfc.predict(X_test)
    return


@app.cell
def _(X_test, pipeline_over_rfc):
    y_pred_proba_over_rfc = pipeline_over_rfc.predict_proba(X_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Consolidação das métricas
    """)
    return


@app.cell
def _(
    pd,
    tml,
    y_pred_grid_rfc,
    y_pred_grid_rfc2,
    y_pred_proba_grid_rfc,
    y_pred_proba_grid_rfc2,
    y_pred_proba_rfc,
    y_pred_rfc,
    y_test,
):
    df_metricas = []

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_rfc, y_pred_proba_rfc, ['RandomForestClassifier']))

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_grid_rfc, y_pred_proba_grid_rfc, ['RandomForest - Grid']))

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_grid_rfc2, y_pred_proba_grid_rfc2, ['RandomForest - Grid 2']))

    #df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_over_rfc, y_pred_proba_over_rfc, ['RandomForest - Grid - Oversample']))

    df_metricas = pd.concat(df_metricas, axis=0)
    df_metricas
    return (df_metricas,)


@app.cell
def _(df_metricas):
    df_metricas.to_clipboard()
    return


if __name__ == "__main__":
    app.run()
