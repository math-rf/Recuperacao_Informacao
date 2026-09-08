from ri_utils import executar_experimentos
import pandas as pd

dataset = ir_datasets.load("cranfield")
# dataset.docs_iter() to access documents -> use loop

print("Iniciando bateria de experimentos...\n")
dfs = []

# Configuração 1: Sem stopwords e sem stemming
print("1. Processando: Texto original (Baseline)...")
dfs.append(executar_experimentos(del_stopwords=False, use_stemming=False))

# Configuração 2: Com remoção de stopwords
print("2. Processando: Apenas remoção de stopwords...")
dfs.append(executar_experimentos(del_stopwords=True, use_stemming=False))

# Configuração 3: Com stemming
print("3. Processando: Apenas stemming...")
dfs.append(executar_experimentos(del_stopwords=False, use_stemming=True))

# Configuração 4: Com remoção de stopwords e stemming
print("4. Processando: Stopwords + Stemming...")
dfs.append(executar_experimentos(del_stopwords=True, use_stemming=True))

# Unindo tudo em um único DataFrame mestre
df_final = pd.concat(dfs, ignore_index=True)

# Salvando o resultado completo para análise
df_final.to_csv('results_default_queries.csv', index=False)

print("\nConcluído! Dimensões do DataFrame final:", df_final.shape)
print(df_final.head())