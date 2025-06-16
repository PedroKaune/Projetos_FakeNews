📰 Classificador de Notícias Falsas com Machine Learning

Descrição
---------
Projeto de classificação de fake news utilizando técnicas de Processamento de Linguagem Natural (NLP) e Machine Learning. 
A proposta é detectar automaticamente se uma notícia é verdadeira ou falsa a partir de seu conteúdo textual.

Tecnologias utilizadas
----------------------
- Python
- Pandas e NumPy para manipulação de dados
- NLTK para pré-processamento de texto (stopwords, lematização)
- Tqdm para barras de progresso durante o processamento
- Scikit-learn para vetorização com TF-IDF e modelo de classificação
- Modelo: PassiveAggressiveClassifier

Etapas do projeto
-----------------
- Carregamento de duas bases de dados: uma com notícias falsas (Fake.csv) e outra com notícias verdadeiras (True.csv)
- Adição de rótulos (label) para cada tipo de notícia
- Junção das bases em um único dataframe
- Combinação do título e corpo da notícia em um único campo de texto
- Limpeza e normalização dos textos (remoção de pontuações, números, stopwords e lematização)
- Separação entre dados de treino e teste
- Vetorização dos textos com TF-IDF
- Treinamento do modelo com Passive Aggressive Classifier
- Avaliação do modelo com métricas como acurácia e matriz de confusão

Resultados esperados
--------------------
- Classificador capaz de distinguir com boa precisão notícias falsas de verdadeiras
- Acurácia geralmente superior a 90%, dependendo da configuração
- Matriz de confusão para análise mais detalhada do desempenho do modelo

Possíveis melhorias
-------------------
- Testar outros modelos como SVM, Random Forest ou BERT
- Implementar uma API para consumo do modelo em tempo real
- Criar uma interface web para testes interativos
- Incluir análise de sentimento ou categorias de temas nas notícias

Objetivo do projeto
-------------------
Demonstrar na prática a aplicação de técnicas de NLP e aprendizado de máquina em um problema real e atual. 
Este projeto é voltado para fins educacionais e de portfólio, com foco na construção de soluções interpretáveis e eficientes para problemas de classificação de texto.
