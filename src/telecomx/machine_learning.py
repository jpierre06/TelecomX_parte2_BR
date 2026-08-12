from typing import Union
import copy

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from scipy.stats import levene
import pingouin as pg

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve
)
from yellowbrick.classifier import ClassificationReport

import utils.local_tools as lt


def graf_correlacao_variaveis(df, colunas_analise, tam_figura:tuple=(8, 8), tam_fonte:int=12):
    corr = df[colunas_analise].corr()

    plt.figure(figsize = tam_figura)
    plt.rcParams.update({'font.size': tam_fonte})
    sns.heatmap(corr, vmin = -1, vmax = 1, center = 0, annot=True, fmt=".2f", square=True, linewidths=.5)
    return plt


def avaliar_modelo(y_test, y_pred, y_pred_proba, print_graph=True):

    print("Métricas:")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    try:
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
    except:
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nRelatório de classificação:\n")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrix de confusão:\n")
    print(cm)

    if print_graph:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Churn','Churn'], yticklabels=['Não Churn','Churn'])
        plt.title('Matriz de Confusão')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.show()
    
        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        except:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Curva ROC')
        plt.legend()
        plt.show()
    
        try:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        except:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.show()


def intervalo_conf(resultado, tipo_teste='test_score'):
    test_score = resultado.get(tipo_teste, 'test_score')
    
    media = test_score.mean()
    desvio_padrao = test_score.std()
    intervalo_confianca = (media - 2* desvio_padrao, min(media + 2* desvio_padrao, 1))
    print(f'Tipo de teste: {tipo_teste}')
    print(f'Média: {media:.4f}\nDesvio padrão: {desvio_padrao:.4f}')
    print(f'Intervalo de confiança entre: {intervalo_confianca[0]:.4f} e {intervalo_confianca[1]:.4f}')


def plotar_classification_report(model, X_train, y_train, X_test, y_test, support=True):
    # Visualizador
    model_copy = copy.deepcopy(model)
    visualizer = ClassificationReport(model_copy, support=True)
    
    # Treinamento e avaliação
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show();    


def df_feature_importances(model, base_100 = True):
    importance = model.feature_importances_
    columns = model.feature_names_in_
    
    df_ = pd.DataFrame({'Feature': columns, 'Importances': importance})
    df_ = df_.sort_values(by=['Importances'], ascending=False).reset_index(drop=True)
    df_['Importance_Cumulative'] = df_['Importances'].cumsum()

    numeric_cols = df_.select_dtypes(include='number')
    df_[numeric_cols.columns] = numeric_cols.round(6)

    if base_100:
        df_[numeric_cols.columns] = numeric_cols * 100

    return df_    


def get_classifier_metrics(y_true, y_predict, y_predict_proba):

    dict_metrics = {}
    dict_functions = {
        'accuracy': accuracy_score, 
        'precision': precision_score, 
        'recall': recall_score, 
        'f1': f1_score, 
    }

    for label, function in dict_functions.items():
        dict_metrics.update({label: function(y_true, y_predict)})

    try:
        roc_auc = roc_auc_score(y_true, y_predict_proba[:, 1])
    except:
        roc_auc = roc_auc_score(y_true, y_predict_proba)
    dict_metrics.update({'roc_auc': roc_auc})
    
    cm = confusion_matrix(y_true, y_predict)
    dict_metrics.update({'support': len(y_true)})
    dict_metrics.update({'True Negative': cm[0][0]})
    dict_metrics.update({'False Positive': cm[0][1]})
    dict_metrics.update({'False Negative': cm[1][0]})
    dict_metrics.update({'True Positive': cm[1][1]})

    return dict_metrics


def df_classifier_metrics(y_true, y_predict, y_predict_proba, name_model):
    classifier_metrics = get_classifier_metrics(y_true, y_predict, y_predict_proba)
    return pd.DataFrame(classifier_metrics, name_model)


def verificar_random_state(obj, nome='objeto'):
    """
    Verifica recursivamente se o parâmetro 'random_state' está definido
    para o objeto fornecido (modelo, pipeline, etc).

    Parameters:
    - obj: modelo sklearn ou imblearn (pode ser um pipeline ou estimador).
    - nome: nome identificador do objeto (usado para mensagens informativas).
    """
    # Caso o objeto seja uma pipeline (sklearn ou imblearn)
    if hasattr(obj, 'steps'):
        for nome_etapa, estimador in obj.steps:
            verificar_random_state(estimador, nome=f"{nome}.{nome_etapa}")
        return True
    else:
        params = obj.get_params()
        if 'random_state' in params:
            if params['random_state'] is None:
                #print(f"[⚠️ Atenção] '{nome}' possui 'random_state=None'. Defina um valor fixo para reprodutibilidade.")
                return False
            else:
                #print(f"[✅ OK] '{nome}' possui 'random_state={params['random_state']}'")
                return True
        else:
            #print(f"[❌ Ausente] '{nome}' não possui o parâmetro 'random_state'.")
            return False


def df_result_selected_features(df_importance, model, X_train, y_train, X_test, y_test, init_result=0, end_result=None, step_result=5, print_step=False):

    model_selected_features = copy.deepcopy(model)
    
    len_features = len(df_importance)
    if not end_result is None:
        len_features = end_result
        
    df_results = pd.DataFrame(index=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'support','True Negative', 'False Positive', 'False Negative', 'True Positive'])
    
    for count in range(init_result, len_features + step_result, step_result):
        count = 1 if count == 0 else count
        selected_features = df_importance.Feature.values[:count]
        
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        if not verificar_random_state(model_selected_features):
            print("Recomenda-se definir 'random_state' no modelo para garantir reprodutibilidade.")
        
        # assert hasattr(model_selected_features, 'random_state') and model.random_state is not None, \
        # "Recomenda-se definir 'random_state' no modelo para garantir reprodutibilidade."

        model_selected_features.fit(X_train_selected, y_train)
        y_pred = model_selected_features.predict(X_test_selected)
        y_pred_proba = model_selected_features.predict_proba(X_test_selected)
    
        metrics = get_classifier_metrics(y_test, y_pred, y_pred_proba)
        count = min(count, len_features)
        df_results[str(count)] = list(metrics.values())
    
        print(f'Selected features - {count}\n') if print_step else None
    
    return df_results


def df_specific_confusion_matrix(X_true:pd.DataFrame, y_true:list, y_predict:list, independent_variable:str, filter_value:Union[str, list]=1, print_only=False):

    try:
        df_true = X_true.copy()
        df_true['target'] = y_true
        df_true['predict'] = y_predict
        
        if isinstance(filter_value, int):
            df_filter = df_true.query(f"`{independent_variable}` == @filter_value")
        elif isinstance(filter_value, list):
            init_value = filter_value[0]
            end_value = filter_value[1]
            df_filter = df_true.query(f"`{independent_variable}`.between(@init_value, @end_value, inclusive='both')")
        else:
            print(f"Invalid type {type(filter_value)} for variable filter_value.")
        
        cm = confusion_matrix(y_true = df_filter.target, y_pred = df_filter.predict)
        tn = cm[0][0] # True Negative
        fp = cm[0][1] # False Negative
        fn = cm[1][0] # False Positive
        tp = cm[1][1] # True Positive
        
        true_prediction = np.trace(cm) # diagonal sum
        accuracy = true_prediction / np.sum(cm)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (2 * precision * recall) / (precision + recall)
        roc_curve = fp / (fp + tn)
        npv = tn / (tn + fn) # negative predictive value
    except:
        return None

    if print_only:
        print(f'\n##### Metrics #####')
        print(f'Independent variable: {independent_variable}')
        print(f'Accuracy: {accuracy:.6f}')
        print(f'Precision: {precision:.6f}')
        print(f'Recall: {recall:.6f}')
        print(f'F1-Score: {f1_score:.6f}')
        print(f'Roc Curve: {roc_curve:.6f}')
        print(f'Negative predictive value: {npv:.6f}')
    
        print(f'\n##### Confusion matrix #####')
        print(cm)
    else:
        data_metrics = {
            'support': np.sum(cm),
            'accuracy': accuracy,
            'negative predictive value': npv,
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            #'roc_curve': roc_curve,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
        }
        df_metrics = pd.DataFrame(data_metrics, index=[independent_variable])
        df_metrics.insert(0, 'filter_value', [filter_value])
    
        return df_metrics
    



def df_final_modelo_v1(df_):
    ...
    df = df_.copy()

    # Deleção coluna ID de cliente
    df.drop(columns=['customerID'], inplace=True)
    # Transformando coluna gender em binária
    df.customer_gender = df.customer_gender.map({'Female': 1, 'Male': 0})

    # Deleção após análise de correlação e fator de inflação
    df.drop(columns=['only_PhoneService', 'only_InternetService', 'both_Phone_InternetService'], inplace=True)
    df.drop(columns=['account_Charges_Daily'], inplace=True)
    df.drop(columns=[
        'internet_OnlineSecurity', 
        'internet_OnlineBackup', 
        'internet_DeviceProtection', 
        'internet_TechSupport', 
        'internet_StreamingTV', 
        'internet_StreamingMovies'
    ], inplace=True)

    # Deleção de colunas binned criado para análise gráfica
    df.drop(columns=['customer_tenure_bins', 'account_Charges_Monthly_bins', 'account_Charges_Total_bins'], inplace=True)

    # Criação de variáveis categorias binned otimizado
    df['tenure_bin'] = lt.optimal_bins(df, 'customer_tenure', 'Churn')
    df['charges_monthly_bin'] = lt.optimal_bins(df, 'account_Charges_Monthly', 'Churn')
    df['charges_total_bin'] = lt.optimal_bins(df, 'account_Charges_Total', 'Churn')

    # Deleção após análise de qui-quadrado
    df.drop(columns=['customer_gender', 'phone_PhoneService'], inplace=True)

    # Deleção após análise de correlação e fator de inflação
    df.drop(columns=['customer_tenure'], inplace=True)
    df.drop(columns=['account_Charges_Monthly'], inplace=True)
    df.drop(columns=['charges_monthly_bin'], inplace=True)
    df.drop(columns=['tenure_bin'], inplace=True)
    df.drop(columns=['internet_InternetService'], inplace=True)
    df.drop(columns=['account_Contract'], inplace=True)

    variaveis_binarias = lt.get_columns_binary(df)
    variaveis_categoricas, dominios = lt.get_columns_categorical(df, 12)
    variaveis_multi_categoricas = list(set(variaveis_categoricas).difference(variaveis_binarias))

    # Tranformação de variáveis multi categóricas com One-Hot Encoding
    df_ohe = pd.get_dummies(df, columns=variaveis_multi_categoricas, drop_first=False, dtype=int)

    # Agregação de PaymentMethod
    df_ohe['account_PaymentMethod_automatic'] = df_ohe['account_PaymentMethod_Bank transfer (automatic)'] + df_ohe['account_PaymentMethod_Credit card (automatic)']
    df_ohe['account_PaymentMethod_check'] = df_ohe['account_PaymentMethod_Electronic check'] + df_ohe['account_PaymentMethod_Mailed check']
    colunas_delecao = [
        'account_PaymentMethod_Bank transfer (automatic)',
        'account_PaymentMethod_Credit card (automatic)',
        'account_PaymentMethod_Electronic check',
        'account_PaymentMethod_Mailed check',
        'internet_Service_Description_No',
        'account_Charges_Total',
    ]
    df_ohe.drop(columns=colunas_delecao, inplace=True)

    df_ohe = lt.adjust_column_names(df_ohe)

    return df_ohe