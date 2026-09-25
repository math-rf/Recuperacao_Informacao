import re
import matplotlib.pyplot as plt
from custom_queries import rel_tag
from ir_utils import carregar_cranfield

def plot_comparacao_queries(
    query_id: str,
    modelo: str,
    query_original: str,
    query_modificada: str,
    top10_original: list,
    top10_modificado: list,
    relevant_docs: dict,
    caminho_saida: str = None,
):
  """Gera uma figura comparando os Top-10 da Query Original vs Modificada para um mesmo modelo.
  Resultados são gerados no arquivo custom_queries_analysis.txt, necessário inputar queries, modelo e top-10s,"""

  def preparar_linhas(docs):
    linhas, cores = [], []
    for rank, doc in enumerate(docs, start=1):
      rel = relevant_docs.get(str(doc), 0)
      linhas.append([str(rank), str(doc), str(rel)])  #[cite: 4]
      # Destaca em verde claro (#d8f3dc) linhas com documentos relevantes
      if rel > 0:
        cores.append(['#d8f3dc'] * 3)  #[cite: 4]
      else:
        cores.append(['#ffffff'] * 3)
    return linhas, cores

  linhas_orig, cores_orig = preparar_linhas(top10_original)
  linhas_mod, cores_mod = preparar_linhas(top10_modificado)
  col_labels = ['#', 'Doc', 'Rel']  #[cite: 4]


  set_orig = set(top10_original)
  set_mod = set(top10_modificado)
  intersecao = len(set_orig & set_mod)
  uniao = len(set_orig | set_mod)

  jaccard_val = (intersecao / uniao)
  

  # Configuração da figura
  fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
  ax.axis('off')

  # Cabeçalho Principal: Nome da Consulta e Modelo avaliado
  plt.text(
      0.5,
      0.97,
      f'{query_id} — Modelo: {modelo}',
      fontsize=13,
      weight='bold',
      ha='center',
      transform=ax.transAxes,
  )  #
  plt.text(
      0.5,
      0.92,
      f'Original: "{query_original}"',
      fontsize=9.5,
      style='italic',
      ha='center',
      transform=ax.transAxes,
  )  #
  plt.text(
      0.5,
      0.87,
      f'Modificada: "{query_modificada}"',
      fontsize=9.5,
      weight='semibold',
      color='#1b4965',
      ha='center',
      transform=ax.transAxes,
  )  #[cite: 5]


  # Tabela 1: Query Original (Esquerda)
  tab_orig = ax.table(
      cellText=linhas_orig,
      colLabels=col_labels,
      cellColours=cores_orig,
      cellLoc='center',
      bbox=[0.02, 0.05, 0.44, 0.70],  #[cite: 4]
  )

  # Tabela 2: Query Modificada (Direita)
  tab_mod = ax.table(
      cellText=linhas_mod,
      colLabels=col_labels,
      cellColours=cores_mod,
      cellLoc='center',
      bbox=[0.54, 0.05, 0.44, 0.70],  #[cite: 4]
  )

  # Título posicionado no centro horizontal da tabela (0.02 + 0.44/2 = 0.24) e acima dela (y=0.78)
  ax.text(
    0.24, 0.78, 
    'Top-10 da Query Original', 
    fontsize=11, 
    weight='bold', 
    ha='center', 
    transform=ax.transAxes
)

  ax.text(
    0.76, 0.78, 
    'Top-10 da Query Modificada', 
    fontsize=11, 
    weight='bold', 
    ha='center', 
    transform=ax.transAxes
)

  # Estilização das bordas e cabeçalhos das tabelas
  for tab in [tab_orig, tab_mod]:
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    for (row, col), cell in tab.get_celld().items():
      cell.set_edgecolor('black')  #[cite: 4]
      cell.set_linewidth(1.0)
      if row == 0:
        cell.set_text_props(weight='semibold')
        cell.set_facecolor('#f0f2f5')

    texto_jaccard = (
      f'Similaridade Jaccard: {jaccard_val:.2f} ({intersecao}/10 documentos em comum)'
  )
  plt.text(
      0.5,
      0.0001,
      texto_jaccard,
      fontsize=10.5,
      weight='bold',
      color='#2b2d42',
      ha='center',
      va='bottom',
      transform=ax.transAxes,
      bbox=dict(
          boxstyle='round,pad=0.4',
          facecolor='#edf2f4',
          edgecolor='#8d99ae',
          linewidth=1.0,
      ),
  )

  plt.tight_layout()

  if caminho_saida:
    plt.savefig(caminho_saida, bbox_inches='tight')
  plt.show()

# O texto é a fonte da identificação; o rótulo abaixo serve apenas para organização.
query_original = (
  "has anyone developed an analysis which accurately establishes the large "
  "deflection behaviour of conical shells ."
)

def normalizar_query(texto):
  return " ".join(re.findall(r"\w+", texto.casefold()))

_, _, _, _, gabarito, consultas = carregar_cranfield(
    del_stopwords=True,
    use_stemming=True,
)
query_id_rel = next(
  query_id
  for query_id, texto, _ in consultas
  if normalizar_query(texto) == normalizar_query(query_original)
)
rel_map = gabarito[query_id_rel]

modelo = "BM25" 

plot_comparacao_queries(
    query_id="Consulta 5", 
    modelo=modelo,
    query_original=query_original,
    query_modificada=(
        "conical conical shells shells analysis"
    ),  
    top10_original= ['930', '1058', '1070', '1059', '1068', '926', '1053', '743', '931', '830'],
    top10_modificado= ['1070', '1059', '1058', '930', '898', '935', '934', '937', '931', '938'],
    relevant_docs=rel_map, 
    caminho_saida=f"comparacao_consulta5_{modelo}.png",
)