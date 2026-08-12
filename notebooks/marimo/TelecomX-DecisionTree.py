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
        DecisionTreeClassifier,
        FeatureImportances,
        NearMiss,
        PrecisionRecallDisplay,
        RocCurveDisplay,
        SMOTE,
        StratifiedKFold,
        average_precision_score,
        confusion_matrix,
        cross_validate,
        imbpipeline,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _():
    import copy

    return (copy,)


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
    ## Treinamento do Modelo - DecisionTreeClassifier
    """)
    return


@app.cell
def _(DecisionTreeClassifier, NUM_SEMENTE_ALEATORIA, X_train, y_train):
    model_dtc = DecisionTreeClassifier(random_state=NUM_SEMENTE_ALEATORIA)
    model_dtc.fit(X_train, y_train)
    return (model_dtc,)


@app.cell
def _(X_test, X_train, model_dtc, y_test, y_train):
    print(f'Acurácia Decision Tree - Treinamento: {model_dtc.score(X_train, y_train)*100:.2f}%')
    print(f'Acurácia Decision Tree - Teste: {model_dtc.score(X_test, y_test)*100:.2f}%')
    return


@app.cell
def _(DecisionTreeClassifier, NUM_SEMENTE_ALEATORIA, X_train, y_train):
    model_dtc_1 = DecisionTreeClassifier(max_depth=10, random_state=NUM_SEMENTE_ALEATORIA)
    model_dtc_1.fit(X_train, y_train)
    return (model_dtc_1,)


@app.cell
def _(X_test, X_train, model_dtc_1, y_test, y_train):
    print(f'Acurácia Decision Tree - Treinamento: {model_dtc_1.score(X_train, y_train) * 100:.2f}%')
    print(f'Acurácia Decision Tree - Teste: {model_dtc_1.score(X_test, y_test) * 100:.2f}%')
    return


@app.cell
def _(X_test, model_dtc_1):
    y_pred_dtc = model_dtc_1.predict(X_test)
    return (y_pred_dtc,)


@app.cell
def _(X_test, model_dtc_1):
    y_pred_proba_dtc = model_dtc_1.predict_proba(X_test)
    return (y_pred_proba_dtc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Matriz de confusão
    """)
    return


@app.cell
def _(confusion_matrix, y_pred_dtc, y_test):
    matriz_confusao = confusion_matrix(y_true = y_test, y_pred = y_pred_dtc)
    return (matriz_confusao,)


@app.cell
def _(matriz_confusao, plt, sns):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Churn','Churn'], yticklabels=['Não Churn','Churn'])
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.title('Matriz de Confusão')
    plt.show()
    return


@app.cell
def _(X_test, lt, pd, tml, y_pred_dtc, y_test):
    list_df = []
    colunas_binarias = lt.identify_columns_binary_values(X_test)
    for c in colunas_binarias:
        list_df.append(tml.df_specific_confusion_matrix(X_test, y_test, y_pred_dtc, c))
    df_independent = pd.concat(list_df, axis=0)

    df_independent
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Curva ROC
    """)
    return


@app.cell
def _(RocCurveDisplay, y_pred_dtc, y_test):
    RocCurveDisplay.from_predictions(y_true = y_test, y_pred = y_pred_dtc, name = 'Decision Tree Classifier');
    return


@app.cell
def _(roc_auc_score, y_pred_dtc, y_test):
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred_dtc):.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Curva de precisão x recall
    """)
    return


@app.cell
def _(PrecisionRecallDisplay, y_pred_dtc, y_test):
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_dtc, name = 'Decision Tree Classifier');
    return


@app.cell
def _(average_precision_score, y_pred_dtc, y_test):
    print(f'Precisão média: {average_precision_score(y_test, y_pred_dtc):.6f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_pred_dtc, y_pred_proba_dtc, y_test):
    tml.df_classifier_metrics(y_test, y_pred_dtc, y_pred_proba_dtc, ['DecisionTree'])
    return


@app.cell
def _(tml, y_pred_dtc, y_pred_proba_dtc, y_test):
    tml.avaliar_modelo(y_test, y_pred_dtc, y_pred_proba_dtc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    📊 1. Métricas principais (agregadas)

    | Métrica      | Valor  | Interpretação                                                                                                                |
    | ------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------- |
    | **Acurácia** | 0.7547 | O modelo acerta \~75,5% das previsões no total.                                                                              |
    | **Precisão** | 0.5323 | Quando o modelo prevê "classe 1", ele acerta \~53,2% das vezes.                                                              |
    | **Recall**   | 0.5022 | O modelo consegue identificar \~50,2% dos verdadeiros positivos (classe 1).                                                  |
    | **F1-Score** | 0.5168 | Harmonia entre precisão e recall: equilíbrio razoável, mas ainda abaixo do ideal.                                            |
    | **ROC AUC**  | 0.7439 | Boa separação entre classes. Indica que o modelo está aprendendo a distinguir as classes (quanto mais próximo de 1, melhor). |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    🧾 2. Relatório de Classificação

    Este relatório apresenta métricas separadas para cada classe:

    Classe 0 (sem churn, por exemplo):

        Precisão: 0.83 → Quando o modelo diz que um cliente vai ficar (classe 0), ele acerta 83% das vezes.

        Recall: 0.84 → De todos os clientes que realmente ficaram, ele identificou 84%.

        F1-score: 0.84 → Excelente desempenho na classe 0.

    Classe 1 (com churn):

        Precisão: 0.53 → Quando o modelo prevê churn, acerta 53% das vezes.

        Recall: 0.50 → De todos os que realmente deram churn, detecta apenas metade.

        F1-score: 0.52 → Baixo desempenho comparado à classe 0.

        ⚠️ O modelo está desequilibrado: ele funciona melhor para a classe 0 do que para a classe 1, que geralmente é a mais importante em casos de churn.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    🔢 3. Matriz de Confusão

    Legenda:

        1098 = Verdadeiros Negativos (VN): previu 0, era 0 → acerto

        203 = Falsos Positivos (FP): previu 1, era 0 → erro

        229 = Falsos Negativos (FN): previu 0, era 1 → erro

        231 = Verdadeiros Positivos (VP): previu 1, era 1 → acerto

    Interpretação:

        Erros em classe 1 (churn) são significativos:

            229 casos o modelo falhou em prever o churn (falsos negativos)

            Isso é crítico, pois você perde a chance de agir para reter esses clientes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    📌 Conclusão

    | Categoria                  | Avaliação                                                                                                |
    | -------------------------- | -------------------------------------------------------------------------------------------------------- |
    | **Desempenho geral**       | Mediano. Boa acurácia, mas mascarada por desequilíbrio entre classes.                                    |
    | **Classe majoritária (0)** | Excelente desempenho.                                                                                    |
    | **Classe minoritária (1)** | Desempenho fraco. O modelo **perde metade dos casos reais** de churn.                                    |
    | **ROC AUC = 0.7439**       | Indica que o modelo **tem potencial**, mas precisa de ajustes para melhorar recall/precisão na classe 1. |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validação Cruzada Estratificada
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Como a variável target está desbalanceada, utilizar StratifiedKFold
    """)
    return


@app.cell
def _(
    LIST_SCORING,
    NUM_SEMENTE_ALEATORIA,
    StratifiedKFold,
    X,
    cross_validate,
    model_dtc_1,
    y,
):
    _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=NUM_SEMENTE_ALEATORIA)
    cv_resultados = cross_validate(model_dtc_1, X, y, cv=_skf, scoring=LIST_SCORING)
    return (cv_resultados,)


@app.cell
def _(LIST_SCORING, cv_resultados, tml):
    for _ls in LIST_SCORING:
        _teste_ls = f'test_{_ls}'
        print('')
        tml.intervalo_conf(cv_resultados, _teste_ls)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Balanceamento de dados
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Pipeline para validação Oversampling
    """)
    return


@app.cell
def _(NUM_SEMENTE_ALEATORIA, SMOTE, imbpipeline, model_dtc_1):
    pipeline_dtc = imbpipeline([('oversample', SMOTE(random_state=NUM_SEMENTE_ALEATORIA)), ('DTC', model_dtc_1)])
    return (pipeline_dtc,)


@app.cell
def _(
    LIST_SCORING,
    NUM_SEMENTE_ALEATORIA,
    StratifiedKFold,
    X,
    cross_validate,
    pipeline_dtc,
    y,
):
    _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=NUM_SEMENTE_ALEATORIA)
    cv_resultados_1 = cross_validate(pipeline_dtc, X, y, cv=_skf, scoring=LIST_SCORING)
    return (cv_resultados_1,)


@app.cell
def _(LIST_SCORING, cv_resultados_1, tml):
    for _ls in LIST_SCORING:
        _teste_ls = f'test_{_ls}'
        print('')
        tml.intervalo_conf(cv_resultados_1, _teste_ls)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Pipeline para validação Undersampling
    """)
    return


@app.cell
def _(NearMiss, imbpipeline, model_dtc_1):
    pipeline_dtc_1 = imbpipeline([('undersample', NearMiss(version=3)), ('DTC', model_dtc_1)])
    return (pipeline_dtc_1,)


@app.cell
def _(
    LIST_SCORING,
    NUM_SEMENTE_ALEATORIA,
    StratifiedKFold,
    X,
    cross_validate,
    pipeline_dtc_1,
    y,
):
    _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=NUM_SEMENTE_ALEATORIA)
    cv_resultados_2 = cross_validate(pipeline_dtc_1, X, y, cv=_skf, scoring=LIST_SCORING)
    return (cv_resultados_2,)


@app.cell
def _(LIST_SCORING, cv_resultados_2, tml):
    for _ls in LIST_SCORING:
        _teste_ls = f'test_{_ls}'
        print('')
        tml.intervalo_conf(cv_resultados_2, _teste_ls)
    return


@app.cell
def _(cv_resultados_2):
    cv_resultados_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Pipeline para validação Oversampling com Decision Tree Classifier
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    O balanceamento de Oversampling apresentou melhor resultado que Undersampling
    """)
    return


@app.cell
def _(NUM_SEMENTE_ALEATORIA, SMOTE, imbpipeline, model_dtc_1):
    pipeline_over_dtc = imbpipeline([('oversample', SMOTE(random_state=NUM_SEMENTE_ALEATORIA)), ('DTC', model_dtc_1)])
    return (pipeline_over_dtc,)


@app.cell
def _(X_train, pipeline_over_dtc, y_train):
    pipeline_over_dtc.fit(X_train, y_train)
    return


@app.cell
def _(X_test, pipeline_over_dtc):
    y_pred_over_dtc = pipeline_over_dtc.predict(X_test)
    return (y_pred_over_dtc,)


@app.cell
def _(X_test, pipeline_over_dtc):
    y_pred_proba_over_dtc = pipeline_over_dtc.predict_proba(X_test)
    return (y_pred_proba_over_dtc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_pred_over_dtc, y_pred_proba_over_dtc, y_test):
    tml.df_classifier_metrics(y_test, y_pred_over_dtc, y_pred_proba_over_dtc, ['DecisioinTree - Oversample'])
    return


@app.cell
def _(tml, y_pred_over_dtc, y_pred_proba_over_dtc, y_test):
    tml.avaliar_modelo(y_test, y_pred_over_dtc, y_pred_proba_over_dtc)
    return


@app.cell
def _(DecisionTreeClassifier, NUM_SEMENTE_ALEATORIA, X_train, y_train):
    model_dtc_2 = DecisionTreeClassifier(max_depth=10, random_state=NUM_SEMENTE_ALEATORIA)
    model_dtc_2.fit(X_train, y_train)
    return (model_dtc_2,)


@app.cell
def _(X_test, X_train, model_dtc_2, tml, y_test, y_train):
    tml.plotar_classification_report(model_dtc_2, X_train, y_train, X_test, y_test)
    return


@app.cell
def _(X_test, X_train, pipeline_over_dtc, tml, y_test, y_train):
    tml.plotar_classification_report(pipeline_over_dtc, X_train, y_train, X_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Seleção de recursos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Identificando importância das features
    """)
    return


@app.cell
def _(FeatureImportances, X_train, copy, model_dtc_2, y_train):
    _viz = FeatureImportances(copy.deepcopy(model_dtc_2))
    _viz.fit(X_train, y_train)
    _viz.show()
    return


@app.cell
def _(FeatureImportances, X_train, copy, model_dtc_2, y_train):
    _viz = FeatureImportances(copy.deepcopy(model_dtc_2), relative=False, topn=15)
    _viz.fit(X_train, y_train)
    _viz.show()
    return


@app.cell
def _(model_dtc_2, tml):
    df_importance = tml.df_feature_importances(model_dtc_2, base_100=True)
    return (df_importance,)


@app.cell
def _(df_importance):
    df_importance
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analisando metricas do modelo de acordo com as features selecionadas
    """)
    return


@app.cell
def _(X_test, X_train, df_importance, model_dtc_2, tml, y_test, y_train):
    tml.df_result_selected_features(df_importance, model_dtc_2, X_train, y_train, X_test, y_test)
    return


@app.cell
def _(X_test, X_train, df_importance, model_dtc_2, tml, y_test, y_train):
    tml.df_result_selected_features(df_importance, model_dtc_2, X_train, y_train, X_test, y_test, init_result=20, end_result=25, step_result=1)
    return


@app.cell
def _(X_test, X_train, df_importance, pipeline_over_dtc, tml, y_test, y_train):
    tml.df_result_selected_features(df_importance, pipeline_over_dtc, X_train, y_train, X_test, y_test)
    return


@app.cell
def _(X_test, X_train, df_importance, pipeline_over_dtc, tml, y_test, y_train):
    tml.df_result_selected_features(df_importance, pipeline_over_dtc, X_train, y_train, X_test, y_test, init_result=20, end_result=25, step_result=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Treinamento DecisionTreeClassifier - Selecionando features
    """)
    return


@app.cell
def _(df_importance):
    selected_features = df_importance.Feature.values[:22]
    return (selected_features,)


@app.cell
def _(X, selected_features):
    X_selected = X[selected_features]
    return (X_selected,)


@app.cell
def _(X_selected):
    X_selected.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Criando novas bases de treino e teste com as features selecionadas
    """)
    return


@app.cell
def _(NUM_SEMENTE_ALEATORIA, TAMANHO_TESTE, X_selected, train_test_split, y):
    X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, random_state=NUM_SEMENTE_ALEATORIA, test_size=TAMANHO_TESTE, stratify=y)
    return X_test_sel, X_train_sel, y_test_sel, y_train_sel


@app.cell
def _(DecisionTreeClassifier, NUM_SEMENTE_ALEATORIA, X_train_sel, y_train_sel):
    model_sel_dtc = DecisionTreeClassifier(max_depth=10, random_state=NUM_SEMENTE_ALEATORIA)
    model_sel_dtc.fit(X_train_sel, y_train_sel)
    return (model_sel_dtc,)


@app.cell
def _(X_test_sel, model_sel_dtc):
    y_sel_pred_dtc = model_sel_dtc.predict(X_test_sel)
    y_sel_pred_proba_dtc = model_sel_dtc.predict_proba(X_test_sel)
    return y_sel_pred_dtc, y_sel_pred_proba_dtc


@app.cell
def _(NUM_SEMENTE_ALEATORIA, SMOTE, imbpipeline, model_sel_dtc):
    pipeline_over_sel_dtc = imbpipeline([
        ('oversample', SMOTE(random_state=NUM_SEMENTE_ALEATORIA)),
        ('DTC', model_sel_dtc),
    ])
    return (pipeline_over_sel_dtc,)


@app.cell
def _(X_train_sel, pipeline_over_sel_dtc, y_train_sel):
    pipeline_over_sel_dtc.fit(X_train_sel, y_train_sel)
    return


@app.cell
def _(X_test_sel, pipeline_over_sel_dtc):
    y_sel_pred_over_dtc = pipeline_over_sel_dtc.predict(X_test_sel)
    y_sel_pred_proba_over_dtc = pipeline_over_sel_dtc.predict_proba(X_test_sel)
    return y_sel_pred_over_dtc, y_sel_pred_proba_over_dtc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relatório de métricas
    """)
    return


@app.cell
def _(tml, y_sel_pred_dtc, y_sel_pred_proba_dtc, y_test_sel):
    tml.df_classifier_metrics(y_test_sel, y_sel_pred_dtc, y_sel_pred_proba_dtc, ['DecisionTree - Selected Feature'])
    return


@app.cell
def _(tml, y_sel_pred_over_dtc, y_sel_pred_proba_over_dtc, y_test_sel):
    tml.df_classifier_metrics(y_test_sel, y_sel_pred_over_dtc, y_sel_pred_proba_over_dtc, ['DecisionTree - Oversample - Selected Feature'])
    return


@app.cell
def _(tml, y_sel_pred_dtc, y_sel_pred_proba_dtc, y_test_sel):
    tml.avaliar_modelo(y_test_sel, y_sel_pred_dtc, y_sel_pred_proba_dtc)
    return


@app.cell
def _(tml, y_sel_pred_over_dtc, y_sel_pred_proba_over_dtc, y_test_sel):
    tml.avaliar_modelo(y_test_sel, y_sel_pred_over_dtc, y_sel_pred_proba_over_dtc)
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
    y_pred_dtc,
    y_pred_over_dtc,
    y_pred_proba_dtc,
    y_pred_proba_over_dtc,
    y_sel_pred_dtc,
    y_sel_pred_over_dtc,
    y_sel_pred_proba_dtc,
    y_sel_pred_proba_over_dtc,
    y_test,
    y_test_sel,
):
    df_metricas = []

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_dtc, y_pred_proba_dtc, ['DecisionTree']))

    df_metricas.append(tml.df_classifier_metrics(y_test, y_pred_over_dtc, y_pred_proba_over_dtc, ['DecisioinTree - Oversample']))

    df_metricas.append(tml.df_classifier_metrics(y_test_sel, y_sel_pred_dtc, y_sel_pred_proba_dtc, ['DecisionTree - Selected Feature']))

    df_metricas.append(tml.df_classifier_metrics(y_test_sel, y_sel_pred_over_dtc, y_sel_pred_proba_over_dtc, ['DecisionTree - Oversample - Selected Feature']))

    df_metricas = pd.concat(df_metricas, axis=0)
    df_metricas
    return (df_metricas,)


@app.cell
def _(df_metricas):
    df_metricas.to_clipboard()
    return


if __name__ == "__main__":
    app.run()
