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

    return GridSearchCV, train_test_split


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
    ## Treinamento do Modelo - CatBoost
    """)
    return


@app.cell
def _():
    from catboost import CatBoostClassifier

    return (CatBoostClassifier,)


@app.cell
def _(CatBoostClassifier, NUM_SEMENTE_ALEATORIA):
    model_cbc = CatBoostClassifier(
        n_estimators=300,
        random_state=NUM_SEMENTE_ALEATORIA
    )
    return (model_cbc,)


@app.cell
def _(X_train, model_cbc, y_train):
    model_cbc.fit(X_train, y_train)
    return


@app.cell
def _(X_test, model_cbc):
    # 8️⃣ Fazer previsões
    y_pred_cbc = model_cbc.predict(X_test)
    y_pred_proba_cbc = model_cbc.predict(X_test)
    return y_pred_cbc, y_pred_proba_cbc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_pred_cbc, y_pred_proba_cbc, y_test):
    tml.avaliar_modelo(y_test, y_pred_cbc, y_pred_proba_cbc, False)
    return


@app.cell
def _(tml, y_pred_cbc, y_pred_proba_cbc, y_test):
    tml.df_classifier_metrics(y_test, y_pred_cbc, y_pred_proba_cbc, ['CatBoost'])
    return


@app.cell
def _(X_test, lt, pd, tml, y_pred_proba_cbc, y_test):
    list_df = []
    colunas_binarias = lt.identify_columns_binary_values(X_test)
    for c in colunas_binarias:
        list_df.append(tml.df_specific_confusion_matrix(X_test, y_test, y_pred_proba_cbc, c))
    df_independent = pd.concat(list_df, axis=0)
    return (df_independent,)


@app.cell
def _(df_independent):
    df_independent.reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Otimizando os hiperparâmetros com o GridSearchCV
    """)
    return


@app.cell
def _():
    param_grid = {
        'depth': [5, 10, 15],               # profundidade da árvore
        'learning_rate': [0.01, 0.1, 0.2],  # taxa de aprendizado
        'iterations': [100, 300, 500],      # número de árvores
        'min_data_in_leaf': [10, 20, 50],   # mínimo de amostras por folha
        'subsample': [0.6, 0.8, 1.0],       # fração de amostras
        'rsm': [0.6, 0.8, 1.0],             # fração de features
        'l2_leaf_reg': [1, 3, 5, 7],        # regularização L2
    }
    return


@app.cell
def _():
    param_grid_1 = {'depth': [5, 10], 'learning_rate': [0.01, 0.1, 0.2], 'iterations': [100, 300], 'min_data_in_leaf': [10, 20], 'subsample': [0.6, 1.0], 'rsm': [0.6, 1.0], 'l2_leaf_reg': [3, 7]}  # profundidade da árvore  # taxa de aprendizado  # número de árvores  # mínimo de amostras por folha  # fração de amostras  # fração de features  # regularização L2
    return (param_grid_1,)


@app.cell
def _(CatBoostClassifier, NUM_SEMENTE_ALEATORIA):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    model_cbc_1 = CatBoostClassifier(random_state=NUM_SEMENTE_ALEATORIA, verbose=0)
    return


@app.cell
def _(CatBoostClassifier, NUM_SEMENTE_ALEATORIA):
    model_cbc_2 = CatBoostClassifier(random_state=NUM_SEMENTE_ALEATORIA, verbose=0)
    return (model_cbc_2,)


@app.cell
def _(GridSearchCV, model_cbc_2, param_grid_1):
    model_grid_cbc = GridSearchCV(estimator=model_cbc_2, param_grid=param_grid_1, scoring='recall', cv=5, n_jobs=-1, verbose=1)
    return (model_grid_cbc,)


@app.cell
def _(X_train, model_grid_cbc, y_train):
    model_grid_cbc.fit(X_train, y_train)
    return


@app.cell
def _(model_grid_cbc):
    model_grid_cbc.best_params_
    return


@app.cell
def _(X_test, model_grid_cbc):
    y_pred_grid_cdc = model_grid_cbc.predict(X_test)
    return (y_pred_grid_cdc,)


@app.cell
def _(X_test, model_grid_cbc):
    y_pred_proba_grid_cdc = model_grid_cbc.predict_proba(X_test)
    return (y_pred_proba_grid_cdc,)


@app.cell
def _(tml, y_pred_grid_cdc, y_pred_proba_grid_cdc, y_test):
    tml.df_classifier_metrics(y_test, y_pred_grid_cdc, y_pred_proba_grid_cdc, ['CatBoost - Grid'])
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
    y_pred_cbc,
    y_pred_grid_cdc,
    y_pred_proba_cbc,
    y_pred_proba_grid_cdc,
    y_test,
):
    df_metricas = []

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_cbc, y_pred_proba_cbc, ['CatBoost']))

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_grid_cdc, y_pred_proba_grid_cdc, ['CatBoost - Grid']))

    df_metricas = pd.concat(df_metricas, axis=0)
    df_metricas
    return (df_metricas,)


@app.cell
def _(df_metricas):
    df_metricas.to_clipboard()
    return


if __name__ == "__main__":
    app.run()
