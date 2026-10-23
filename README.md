# Estudos de Machine Learning na prática

Repositório com os exercícios práticos que usei para aprender, na prática, os conceitos básicos de aprendizado não supervisionado, NLP e visão computacional. Cada exercício usa um dataset real e foca em entender o porquê de cada etapa, e não só em rodar o código.

## Estrutura

```
Projetos-Machine-Learning/
├── 01/   Aprendizado não supervisionado
│   ├── data/
│   ├── k-means.ipynb
│   ├── pca.ipynb
│   ├── anomalias.ipynb
│   ├── tfidf.ipynb
│   ├── rfm.ipynb
│   └── pipeline.ipynb
├── 02/   Processamento de linguagem natural
│   ├── data/
│   ├── nlp.ipynb
│   └── ner.ipynb
├── 03/   Visão computacional
│   ├── data/
│   ├── cv.ipynb
│   └── cv-dogs-cats.ipynb
└── README.md
```

## Conteúdo

### 01. Aprendizado não supervisionado

| Notebook | Dataset | O que pratiquei |
|----------|---------|-----------------|
| [`k-means`](01/k-means.ipynb) | Mall Customers | K-Means, método do cotovelo, escalonamento com `StandardScaler` (e a diferença entre `normalize` e `StandardScaler`) |
| [`pca`](01/pca.ipynb) | Country Data (167 países, 9 features) | PCA, variância explicada, projeção em 2D, K-Means sobre os componentes |
| [`anomalias`](01/anomalias.ipynb) | Credit Card Fraud Detection | Isolation Forest, Local Outlier Factor, avaliação com precision/recall em dados muito desbalanceados |
| [`tfidf`](01/tfidf.ipynb) | BBC News | TF-IDF, dendrograma, `AgglomerativeClustering`, comparação dos clusters com as categorias reais (`crosstab`) |
| [`rfm`](01/rfm.ipynb) | Online Retail | Análise RFM (Recência, Frequência, Valor Monetário), K-Means, recomendação por diferença de conjuntos |
| [`pipeline`](01/pipeline.ipynb) | Credit Card Dataset for Clustering | `StandardScaler` + PCA (escolha de componentes pela variância acumulada) + K-Means, validação com silhouette e Davies-Bouldin |

### 02. Processamento de linguagem natural

| Notebook | Dataset | O que pratiquei |
|----------|---------|-----------------|
| [`nlp`](02/nlp.ipynb) | IMDB 50K Movie Reviews | Análise de sentimento: limpeza de texto (HTML, minúsculas, pontuação, stopwords), TF-IDF sem vazamento entre treino e teste, Regressão Logística |
| [`ner`](02/ner.ipynb) | News Articles Classification | Reconhecimento de entidades nomeadas com spaCy, leitura crítica dos erros do modelo pré-treinado |

### 03. Visão computacional

| Notebook | Dataset | O que pratiquei |
|----------|---------|-----------------|
| [`cv`](03/cv.ipynb) | MNIST | CNN: convolução, pooling, one-hot encoding, treino com Keras, validação vs. teste |
| [`cv-dogs-cats`](03/cv-dogs-cats.ipynb) | Cat and Dog | `ImageDataGenerator`, data augmentation (rotação e flip só no treino), `flow_from_directory`, overfitting |

## Resultados e aprendizados

- **K-Means:** a escala das features muda o resultado. Com `StandardScaler`, os 5 grupos do Mall Customers ficaram bem separados, e com `normalize` (que escala por linha) ficaram misturados.
- **PCA:** 2 componentes capturaram cerca de 63% da variância do dataset de países. Dois componentes servem para visualizar, mas o número usado no clustering deve vir da variância acumulada (7 componentes no pipeline).
- **Detecção de anomalias:** Isolation Forest e LOF tiveram desempenho fraco na fraude de cartão. Com `contamination` exato (0,17%), o recall foi de cerca de 25%, e o LOF não detectou nenhuma fraude. Acurácia alta engana em dados desbalanceados.
- **Clustering de texto:** esporte e entretenimento formaram clusters bem puros, enquanto negócios e política se misturaram por compartilharem vocabulário.
- **Validação de clusters:** no pipeline completo, silhouette de 0,25 e Davies-Bouldin de 1,32 indicaram sobreposição entre os clusters. Testar k=3 não trouxe melhoria clara, porque as duas métricas discordaram.
- **Sentimento:** cerca de 90% de acurácia com TF-IDF e Regressão Logística. O vetorizador é ajustado (`fit`) só no treino.
- **NER:** o modelo pré-treinado erra com siglas ambíguas e com categorias que não existem no treino dele.
- **CNN no MNIST:** cerca de 98% de acurácia no conjunto de teste.
- **Cães e gatos:** cerca de 80% de acurácia, com treino e validação próximos (sem overfitting). O resultado sugere um modelo ainda pequeno ou pouco treinado.

## Datasets

Os datasets não estão no repositório (a pasta `data/` está no `.gitignore`). Baixe cada um e coloque na pasta indicada, ajustando os caminhos nos notebooks se necessário.

**Módulo 01** (`01/data/`)

| Arquivo | Dataset |
|---------|---------|
| `Mall_Customers.csv` | [Mall Customers](https://github.com/subhadeep-123/KMeans-and-Agglomerative-Clustering-on-Mall-Customers-dataset) |
| `Country-data.csv` | [Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) |
| `creditcard.csv` | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `bbc-text.csv` | [BBC News](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category) |
| `ecommerce.csv` | [Online Retail](https://www.kaggle.com/datasets/carrie1/ecommerce-data) |
| `CC GENERAL.csv` | [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) |

**Módulo 02** (`02/data/`)

| Arquivo | Dataset |
|---------|---------|
| `IMDB Dataset.csv` | [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| `ner_data/` | [News Articles Classification](https://www.kaggle.com/datasets/banuprakashv/news-articles-classification-dataset-for-nlp-and-ml) (um CSV por categoria; o notebook usa o de educação) |

**Módulo 03** (`03/data/`)

| Pasta | Dataset |
|-------|---------|
| `dogs-cats/` (com `training_set/` e `test_set/`, cada uma com as subpastas `cats/` e `dogs/`) | [Cat and Dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog) |
| (embutido) | MNIST, carregado direto pelo Keras com `mnist.load_data()` |

## Tecnologias

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, SciPy, NLTK, spaCy, TensorFlow/Keras.

## Como executar

1. Clone o repositório e crie um ambiente virtual.
2. Instale as dependências: `pip install pandas numpy matplotlib seaborn scikit-learn scipy nltk spacy tensorflow`
3. Para o NER, baixe o modelo do spaCy: `python -m spacy download en_core_web_sm`
4. Para as stopwords, rode uma vez `nltk.download('stopwords')`.
5. Baixe os datasets pelos links acima e coloque cada um na pasta `data/` do módulo.
6. Abra o notebook desejado e execute as células em ordem.