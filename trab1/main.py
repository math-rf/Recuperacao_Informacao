from ir_utils import executar_experimentos
import ir_datasets
import pandas as pd
import os

dataset = ir_datasets.load("cranfield")
# dataset.docs_iter() to access documents -> use loop

folder = "Recuperacao_Informacao/trab1/results/"
os.makedirs(folder, exist_ok=True)

print("Iniciando bateria de experimentos...\n")
dfs_map = []
dfs_experimentos = []

# Configuração 1: Sem stopwords e sem stemming
print("1. Processando: Texto original (Baseline)...")
df_exp, df_map = executar_experimentos(del_stopwords=False, use_stemming=False)
dfs_experimentos.append(df_exp)
dfs_map.append(df_map)

# Configuração 2: Com remoção de stopwords
print("2. Processando: Apenas remoção de stopwords...")
df_exp, df_map = executar_experimentos(del_stopwords=True, use_stemming=False)
dfs_experimentos.append(df_exp)
dfs_map.append(df_map)

# Configuração 3: Com stemming
print("3. Processando: Apenas stemming...")
df_exp, df_map = executar_experimentos(del_stopwords=False, use_stemming=True)
dfs_experimentos.append(df_exp)
dfs_map.append(df_map)

# Configuração 4: Com remoção de stopwords e stemming
print("4. Processando: Stopwords + Stemming...")
df_exp, df_map = executar_experimentos(del_stopwords=True, use_stemming=True)
dfs_experimentos.append(df_exp)
dfs_map.append(df_map)

# Unindo tudo em um único DataFrame mestre
df_final_exp = pd.concat(dfs_experimentos, ignore_index=True)
df_final_map = pd.concat(dfs_map, ignore_index=True)

# Salvando o resultado completo para análise
df_final_exp.to_csv(folder + 'results_default_queries.csv', index=False)
df_final_map.to_csv(folder + 'results_default_queries_map.csv', index=False)

print("\nConcluído! Dimensões do DataFrame final (exp e map):", df_final_exp.shape, df_final_map.shape)
