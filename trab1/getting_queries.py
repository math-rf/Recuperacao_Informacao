import ir_datasets
import pandas as pd

# Salvando queries em um csv para análises posteriores

dataset = ir_datasets.load("cranfield")

lista_consultas = []
for query in dataset.queries_iter():
    lista_consultas.append({
        'Query_ID': query.query_id,
        'Texto_Consulta': query.text.strip() # .strip() remove quebras de linha extras
    })

df_consultas = pd.DataFrame(lista_consultas)
df_consultas.to_csv('cranfield_queries_texts.csv', index=False)

print("Arquivo salvo com sucesso! Dimensões:", df_consultas.shape)
print(df_consultas.head())