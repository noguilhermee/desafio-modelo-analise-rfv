# 📊 Modelo de Análise RFV com Clusterização

## 📌 Descrição do Projeto

Este projeto tem como objetivo desenvolver um modelo de **clusterização para segmentação de clientes de um e-commerce**, utilizando dados transacionais reais. A partir da análise do comportamento de compra, são identificados **perfis distintos de clientes**, permitindo apoiar estratégias de **segmentação e personalização de campanhas de marketing**.

O projeto foi desenvolvido como parte da disciplina **Modelos de Clusterização** do curso de **Data Science – DNC**.

---

## 🎯 Objetivos

- Realizar análise exploratória de dados transacionais;
- Executar o pré-processamento dos dados;
- Aplicar algoritmos de clusterização;
- Avaliar a qualidade dos clusters por métricas estatísticas;
- Interpretar os clusters obtidos;
- Propor ações estratégicas com base nos resultados.

---

## 🗂️ Estrutura do Projeto

```text
desafio-modelo-analise-rfv/
│
├── app/
│   ├── main.ipynb
│   └── RID214136_Desafio07.ipynb
│
├── data/
│   └── data.csv
│
├── functions/
│   └── function.py
│
├── .gitignore
└── README.md
```

## 🧠 Metodologia

O desenvolvimento do projeto seguiu as etapas abaixo:

1. **Análise Exploratória dos Dados**
   - Estatísticas descritivas
   - Análise de distribuições
   - Identificação de dados nulos, duplicados, outliers e inconsistências

2. **Pré-processamento**
   - Tratamento de registros inválidos
   - Remoção de duplicatas e outliers
   - Normalização das variáveis numéricas

3. **Clusterização**
   - Aplicação do algoritmo **K-Means**
   - Definição do número ideal de clusters
   - Avaliação com as métricas:
     - Inertia (WCSS)
     - Silhouette Score
     - Davies-Bouldin Score
     - Calinski-Harabasz Score

4. **Análise dos Clusters**
   - Visualização gráfica dos agrupamentos
   - Estatísticas descritivas por cluster
   - Definição de perfis de clientes

5. **Interpretação e Recomendações**
   - Segmentação de clientes
   - Sugestão de ações de marketing direcionadas

---

## 📊 Principais Resultados

- Identificação de **três clusters distintos** de clientes;
- Segmentação baseada principalmente no **valor monetário das compras**;
- Definição dos seguintes perfis:
  - Clientes de baixo valor de compra;
  - Clientes de valor intermediário;
  - Clientes de alto valor de compra.

---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## ▶️ Como Executar

1. Clone o repositório:
   ```bash
   git clone <url-do-repositorio>
