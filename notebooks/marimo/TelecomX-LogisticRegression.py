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
    import matplotlib.pyplot as plt

    return (plt,)


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
        LogisticRegression,
        SMOTE,
        StandardScaler,
        StratifiedKFold,
        cross_validate,
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
    import telecomx.machine_learning as tml

    return (tml,)


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
    return MAXIMO_ITERACAO, NUM_SEMENTE_ALEATORIA, TAMANHO_TESTE


@app.cell
def _():
    LIST_SCORING = ['accuracy','recall', 'precision', 'f1']
    return (LIST_SCORING,)


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
    ## Treinamento do Modelo - LogisticRegression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exige padronização dos dados
    """)
    return


@app.cell
def _(
    LogisticRegression,
    MAXIMO_ITERACAO,
    NUM_SEMENTE_ALEATORIA,
    StandardScaler,
    imbpipeline,
):
    pipeline_lr = imbpipeline([
        ('Scaler', StandardScaler()),
        ('LogisticRegression', LogisticRegression(max_iter=MAXIMO_ITERACAO, random_state=NUM_SEMENTE_ALEATORIA))
    ])
    return (pipeline_lr,)


@app.cell
def _(
    LIST_SCORING,
    NUM_SEMENTE_ALEATORIA,
    StratifiedKFold,
    X,
    cross_validate,
    pipeline_lr,
    y,
):
    skf = StratifiedKFold(n_splits= 5, shuffle= True, random_state= NUM_SEMENTE_ALEATORIA)
    cv_resultados = cross_validate(pipeline_lr, X, y, cv= skf, scoring= LIST_SCORING)
    return (cv_resultados,)


@app.cell
def _(LIST_SCORING, cv_resultados, tml):
    for ls in LIST_SCORING:
        teste_ls = f'test_{ls}'
        print('')
        tml.intervalo_conf(cv_resultados, teste_ls)
    return


@app.cell
def _(cv_resultados):
    cv_resultados
    return


@app.cell
def _(X_train, pipeline_lr, y_train):
    pipeline_lr.fit(X_train, y_train)
    return


@app.cell
def _(X_test, pipeline_lr):
    y_pred_pipe_lr = pipeline_lr.predict(X_test)
    y_pred_proba_pipe_lr = pipeline_lr.predict_proba(X_test)
    return y_pred_pipe_lr, y_pred_proba_pipe_lr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_pred_pipe_lr, y_pred_proba_pipe_lr, y_test):
    tml.avaliar_modelo(y_test, y_pred_pipe_lr, y_pred_proba_pipe_lr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Importância de features
    """)
    return


@app.cell
def _(pipeline_lr):
    pipeline_lr.named_steps.get('LogisticRegression')
    return


@app.cell
def _(pipeline_lr):
    # Acessando o modelo treinado dentro do pipeline
    logreg_model = pipeline_lr.named_steps.get('LogisticRegression')
    return (logreg_model,)


@app.cell
def _(X_train, logreg_model, np, pd):
    # Supondo que X_train é um DataFrame
    coeficientes = logreg_model.coef_[0]  # Para binário, só há uma linha

    # Criar um DataFrame com as features e seus coeficientes
    df_importance_lr = pd.DataFrame({
        'Feature': X_train.columns,
        'Coeficient': coeficientes,
        'Absolute_importance': np.abs(coeficientes)
    })

    # Ordenar pela importância absoluta
    df_importance_lr = df_importance_lr.sort_values(by='Absolute_importance', ascending=False)
    return (df_importance_lr,)


@app.cell
def _(df_importance_lr):
    df_importance_lr
    return


@app.cell
def _(df_importance_lr, plt):
    plt.figure(figsize=(10,6))
    plt.barh(df_importance_lr.Feature, df_importance_lr.Coeficient, color='skyblue')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Coeficiente')
    plt.title('Importância das Features (Logistic Regression)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Otimização de hiperparâmetros
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Otimizando os hiperparâmetros com o GridSearchCV
    """)
    return


@app.cell
def _(logreg_model):
    logreg_model.get_params().keys()
    return


@app.cell
def _():
    param_grid = {
        'LogisticRegression__C': [0.01, 0.1, 1, 10, 100],
        'LogisticRegression__penalty': ['l1', 'l2'],
        'LogisticRegression__class_weight': [None, 'balanced'],
    }
    return (param_grid,)


@app.cell
def _(GridSearchCV, param_grid, pipeline_lr):
    model_grid_lr = GridSearchCV(
        estimator=pipeline_lr,
        param_grid=param_grid,
        cv=5,
        scoring='recall',  # ou 'f1', 'roc_auc'
        n_jobs=-1,
        verbose=2
    )
    return (model_grid_lr,)


@app.cell
def _(X_train, model_grid_lr, y_train):
    model_grid_lr.fit(X_train, y_train)
    return


@app.cell
def _(model_grid_lr):
    model_grid_lr.best_params_
    return


@app.cell
def _(model_grid_lr):
    model_grid_lr.score
    return


@app.cell
def _(X_test, model_grid_lr):
    y_pred_grid_lr = model_grid_lr.predict(X_test)
    return (y_pred_grid_lr,)


@app.cell
def _(X_test, model_grid_lr):
    y_pred_proba_grid_lr = model_grid_lr.predict_proba(X_test)
    return (y_pred_proba_grid_lr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_pred_pipe_lr, y_pred_proba_pipe_lr, y_test):
    tml.df_classifier_metrics(y_test, y_pred_pipe_lr, y_pred_proba_pipe_lr, ['LR - StandardScaler'])
    return


@app.cell
def _(tml, y_pred_grid_lr, y_pred_proba_grid_lr, y_test):
    tml.df_classifier_metrics(y_test, y_pred_grid_lr, y_pred_proba_grid_lr, ['LR - StandardScaler - Grid'])
    return


@app.cell
def _(tml, y_pred_pipe_lr, y_pred_proba_pipe_lr, y_test):
    tml.avaliar_modelo(y_test, y_pred_pipe_lr, y_pred_proba_pipe_lr, False)
    return


@app.cell
def _(tml, y_pred_grid_lr, y_pred_proba_grid_lr, y_test):
    tml.avaliar_modelo(y_test, y_pred_grid_lr, y_pred_proba_grid_lr, False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Balanceamento com Oversampling
    """)
    return


@app.cell
def _(
    LogisticRegression,
    MAXIMO_ITERACAO,
    NUM_SEMENTE_ALEATORIA,
    SMOTE,
    StandardScaler,
    imbpipeline,
):
    pipeline_blover_lr = imbpipeline([
        ('oversample', SMOTE(random_state=NUM_SEMENTE_ALEATORIA)),
        ('Scaler', StandardScaler()),
        ('LogisticRegression', LogisticRegression(max_iter=MAXIMO_ITERACAO, random_state=NUM_SEMENTE_ALEATORIA))
    ])
    return (pipeline_blover_lr,)


@app.cell
def _(GridSearchCV, param_grid, pipeline_blover_lr):
    model_grid_blover_lr = GridSearchCV(
        estimator=pipeline_blover_lr,
        param_grid=param_grid,
        cv=5,
        scoring='recall',  # ou 'f1', 'roc_auc'
        n_jobs=-1,
        verbose=2
    )
    return (model_grid_blover_lr,)


@app.cell
def _(X_train, model_grid_blover_lr, y_train):
    model_grid_blover_lr.fit(X_train, y_train)
    return


@app.cell
def _(model_grid_lr):
    model_grid_lr.best_params_
    return


@app.cell
def _(model_grid_blover_lr):
    model_grid_blover_lr.best_params_
    return


@app.cell
def _(X_test, model_grid_blover_lr):
    y_pred_grid_blover_lr = model_grid_blover_lr.predict(X_test)
    return (y_pred_grid_blover_lr,)


@app.cell
def _(X_test, model_grid_blover_lr):
    y_pred_proba_grid_blover_lr = model_grid_blover_lr.predict_proba(X_test)
    return (y_pred_proba_grid_blover_lr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_pred_grid_blover_lr, y_pred_proba_grid_blover_lr, y_test):
    tml.df_classifier_metrics(y_test, y_pred_grid_blover_lr, y_pred_proba_grid_blover_lr, ['LR - StandardScaler - Grid - Oversampling'])
    return


@app.cell
def _(tml, y_pred_grid_blover_lr, y_pred_proba_grid_blover_lr, y_test):
    tml.avaliar_modelo(y_test, y_pred_grid_blover_lr, y_pred_proba_grid_blover_lr, False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Consolidação das métricas
    """)
    return


@app.cell
def _(
    tml,
    y_pred_grid_blover_lr,
    y_pred_grid_lr,
    y_pred_pipe_lr,
    y_pred_proba_grid_blover_lr,
    y_pred_proba_grid_lr,
    y_pred_proba_pipe_lr,
    y_test,
):
    df_metricas = []

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_pipe_lr, y_pred_proba_pipe_lr, ['LR - StandardScaler']))

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_grid_lr, y_pred_proba_grid_lr, ['LR - StandardScaler - Grid']))

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_grid_blover_lr, y_pred_proba_grid_blover_lr, ['LR - StandardScaler - Grid - Oversampling']))
    return (df_metricas,)


@app.cell
def _(df_metricas, pd):
    df_metricas_1 = pd.concat(df_metricas, axis=0)
    df_metricas_1
    return (df_metricas_1,)


@app.cell
def _(df_metricas_1):
    df_metricas_1.to_clipboard()
    return


if __name__ == "__main__":
    app.run()
