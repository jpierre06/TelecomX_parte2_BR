# 🎓 O projeto faz parte de um Challenge do Curso de Machine Learning da Alura

Challenge do Curso de Machine Learning faz parte da formação de Data Science da Alura em parceria com a ONE (Oracle Next Education)

<br>

# 📊 Projeto de Predição de Churn - TelecomX

![Python](https://img.shields.io/badge/Python-3.14.5-blue)
![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![marimo](https://img.shields.io/badge/marimo-%3E%3D0.23.16-F5D90A)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9.0-blueviolet)
![Pandas](https://img.shields.io/badge/Pandas-3.0.5-yellow)
![NumPy](https://img.shields.io/badge/NumPy-2.5.2-013243)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.11.1-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-4c72b0)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.14.2-orange)

Este projeto tem como objetivo desenvolver um modelo preditivo de **Churn** (evasão de clientes) para uma operadora de telecomunicações, utilizando técnicas de Machine Learning. A análise foi realizada em um ambiente Jupyter Notebook, com foco em reutilização de código, padronização de processos e interpretabilidade dos resultados.

---

## 📌 Sumário

1. [Sobre o Projeto](#sobre-o-projeto)
2. [Objetivo](#objetivo)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Requisitos e Dependências](#requisitos-e-dependências)
5. [Como Executar](#como-executar)
6. [Análise Exploratória (EDA)](#análise-exploratória-eda)
7. [Modelagem de Machine Learning](#modelagem-de-machine-learning)
8. [Scripts Personalizados](#scripts-personalizados)
9. [Métricas de Avaliação](#métricas-de-avaliação)
10. [Contato](#contato)

---

## 📌 Sobre o Projeto

O projeto **TelecomX** visa identificar padrões de comportamento que levam à evasão de clientes, permitindo que a empresa antecipe e aja com estratégias de retenção mais eficazes. A base de dados contém informações sobre clientes, como tempo de contrato, serviços contratados, gastos mensais, dados demográficos e status de churn.

---

## 🎯 Objetivo

Desenvolver um modelo de classificação capaz de prever com alta precisão quais clientes têm maior probabilidade de cancelar seus serviços (churn), utilizando algoritmos de Machine Learning supervisionado.

---

## 🗂️ Estrutura do Projeto
```bash
TelecomX_parte2_BR/
│
├── dados/
│ └── dados_tratados.csv # Dados utilizados no projeto
│
├── src
├──  __pycache__
├──  telecomx
│     ├── __init__.py
│     ├── analysis.py
│     └── machine_learning.py
├──  telecomx_2.egg-info
│     ├── PKG-INFO
│     ├── SOURCES.txt
│     ├── dependency_links.txt
│     ├── requires.txt
│     └── top_level.txt
└──  utils
     ├── __init__.py
     └── local_tools.py	
│
├── notebooks/
├──  jupyter
│    ├── TelecomX-CatBoost.ipynb
│    ├── TelecomX-DecisionTree.ipynb
│    ├── TelecomX-LightGBM.ipynb
│    ├── TelecomX-LogisticRegression.ipynb
│    ├── TelecomX-RandomForest.ipynb
│    ├── TelecomX-XGBoost.ipynb
│    └── TelecomX.ipynb # Notebook principal com EDA e modelagem
└──   marimo
│
├── README.md # Este arquivo
├── python-version # Versão Python - UV
├── pyproject.toml # Configuração de gerenciamento do projeto - UV
├── requirements.txt # Dependências do projeto
└── uv.lock # Dependências do projeto - UV
```

---

## ⚙️ Requisitos e Dependências

### Versão do Python
- **Python 3.14.5**

### Bibliotecas Principais
- `pandas`, `numpy` – manipulação de dados
- `matplotlib`, `seaborn` – visualização
- `scikit-learn` – modelagem ML
- `imblearn` – balanceamento de classes (SMOTE, NearMiss)
- `yellowbrick` – visualização de modelos
- `pingouin`, `scipy` – estatística inferencial
- `optbinning` – binning otimizado
- `jupyter` – ambiente de desenvolvimento
- `marimo` – ambiente de desenvolvimento

<br>

## ▶️ Como Executar

1. Clone o repositório

```bash
git clone https://github.com/jpierre06/TelecomX_parte2_BR.git
cd TelecomX_parte2_BR
```

2. Instale as dependências com o [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```


3. Inicie o Jupyter Notebook:

```bash
uv run jupyter notebook
```

4. Abra o notebook principal:

* Utilizando Jupyterlab
  * Navegue até notebooks/jupyter/TelecomX.ipynb e execute as células em ordem.

* Utilizando marimo
  * Navegue até notebooks/marimo/TelecomX.py e execute as células em ordem.

5. (Opcional) Teste outros modelos:

* Os notebooks individuais (TelecomX-\*.ipynb e TelecomX-\*.py) permitem comparar diferentes algoritmos de ML.

<br>

## 🔍 Análise Exploratória (EDA)

O notebook principal realiza uma análise exploratória detalhada, incluindo:

* Distribuição da variável alvo (Churn)

* Análise de variáveis categóricas e numéricas
* Visualizações com seaborn e matplotlib
* Testes estatísticos:
	* **Teste Qui-Quadrado** para variáveis categóricas
	* **ANOVA e Welch ANOVA** para comparação de médias
	* **Teste de Levene** para homocedasticidade
* Identificação de multicolinearidade com **VIF (Variance Inflation Factor)**
* Criação de variáveis derivadas e **binned otimizados** com `optbinning`

<br>

## 🤖 **Modelagem de Machine Learning**

### **Algoritmos Testados**
* Decision Tree
* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost

### **Estratégias Aplicadas**

* **Balanceamento de classes** com SMOTE e NearMiss
* **Validação cruzada estratificada** (StratifiedKFold)
* **Otimização de hiperparâmetros** com `GridSearchCV`
* **Padronização** com `StandardScaler` (para modelos sensíveis à escala)
* **Pipeline completo** com `imblearn.Pipeline`

### Métricas Utilizadas

* Acurácia, Precisão, Recall, F1-Score
* ROC AUC, Curva ROC e Precision-Recall
* Matriz de Confusão
* Relatório de classificação detalhado

<br>

## 🧩 Scripts Personalizados

Para modularizar e reutilizar o código, foram criados três scripts:

`src/util/local_tools.py`

* Funções genéricas de apoio:
	* *`get_variance_inflation_factor()`*: cálculo de VIF
	* *`get_chi_square():`*` teste qui-quadrado
	* *`get_ttest_ind():`* teste t de médias
	* *`optimal_bins():`* criação de bins otimizados por churn
	* *`apply_percent_category():`* cálculo de percentuais por categoria

<br>

`src/telecomx/analysis.py`

* Funções específicas para análise da base TelecomX:

	* *`carregar_dados_telecomx_normalizado():`* carregamento de dados JSON
	* *`tratar_valores_invalidados():`* limpeza de dados nulos/inválidos
	* *`criar_colunas_derivadas():`* criação de features como additional_InternetService
	* Funções de visualização: gráficos de barras, boxplots, KDE, matriz de correlação

<br>

`src/telecomx/machine_learning.py`

* Funções para avaliação de modelos:
	* *`avaliar_modelo():`* exibe métricas completas (acurácia, recall, F1, etc.)
	* *`df_classifier_metrics():`* retorna DataFrame com métricas
	* *`df_feature_importances():`* importância de features em modelos baseados em árvores
	* *`graf_correlacao_variaveis():`* heatmap de correlação
	* *`plotar_classification_report():`* relatório visual com Yellowbrick


<br>

## 📊 Métricas de Avaliação

As métricas foram escolhidas com base no impacto de negócio:

| Métrica  | Prioridade | Justificativa                                        |
| -------- | ---------- | ----------------------------------------------------- |
| Recall   | Alta       | Identificar a maior parte dos clientes que vão sair |
| Precisão | Média      | Evitar campanhas de retenção desnecessárias         |
| F1-Score | Alta       | Equilíbrio entre recall e precisão                  |
| ROC AUC  | Média      | Avaliação geral do poder discriminativo             |

📌 Dica de negócio: Se o custo de retenção for alto, priorize precisão. Se a perda de clientes for crítica, priorize recall. 

<br>


## 📬 Contato

Se você tiver dúvidas, sugestões ou quiser contribuir com melhorias, sinta-se à vontade para entrar em contato ou abrir uma issue no repositório.

### 🛠️ Desenvolvido por: Jean Pierre
### 📧 jps.data.analise@gmail.com
### 💼 https://www.linkedin.com/in/jeanpierresantana
