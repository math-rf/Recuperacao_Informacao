RECUPERACAO DA INFORMACAO - TRABALHO 1


i) INTEGRANTES

Gustavo Cardozo De Moraes Moreira
Matheus Rodrigues Ferreira

ii) VERSAO DA LINGUAGEM E PRINCIPAIS BIBLIOTECAS

Python 3.14

- ir_datasets   : download e leitura da base Cranfield
- nltk          : tokenizacao, stopwords e stemming (Snowball, ingles)
- scikit-learn  : TfidfVectorizer, CountVectorizer, cosine_similarity
- numpy         : calculos numericos
- pandas        : manipulacao de dados e CSV
- matplotlib    : graficos


BASE DE DADOS

Colecao Cranfield: resumos de artigos de aerodinamica/aeronautica, em ingles.
1.400 documentos e 225 consultas, com julgamentos de relevancia (qrels). A base e baixada
automaticamente pela biblioteca ir_datasets na primeira execucao

INSTALACAO DAS DEPENDENCIAS

O trabalho foi produzido no ambiente do python 3.14

Instalar as bibliotecas:

    pip install -r requirements.txt


iv) EXECUCAO

Execute sempre de dentro da pasta trab1:

    cd trab1

Rode os scripts nesta ordem:

    python main.py                 (roda os modelos nas 4 configuracoes de
                                    pre-processamento; gera os resultados)
    python getting_queries.py      (salva o texto das consultas)
    python comparacao_modelos.py   (MAP agregado; precisa do main.py)
    python query_analysis.py       (melhores/piores consultas; precisa do
                                    main.py e do getting_queries.py)
    python bm25_params_test.py     (variacao dos parametros k1 e b do BM25)
    python error_analysis.py       (analise de erros)
    python custom_queries.py       (consultas originais x modificadas)

Os resultados (CSV, graficos e relatorios) sao salvos em trab1/results/.
Rodar novamente sobrescreve os arquivos existentes.