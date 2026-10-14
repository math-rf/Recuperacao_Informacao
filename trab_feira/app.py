import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(
    page_title="Como Funciona um Motor de Busca?",
    page_icon="🔍",
    layout="wide"
)

# Base de dados de exemplo (Corpus)
@st.cache_data
def carregar_corpus():
    return [
        {"id": 1, "titulo": "Bolo de Cenoura Fofinho", "texto": "Aprenda a fazer um bolo de cenoura fofinho com cobertura de chocolate cremoso. Uma receita fácil e rápida para o café da tarde."},
        {"id": 2, "titulo": "Panqueca de Frango com Requeijão", "texto": "Receita prática de massa de panqueca salgada recheada com frango desfiado e requeijão cremoso ao forno."},
        {"id": 3, "titulo": "Brigadeiro Gourmet de Chocolate", "texto": "O clássico brigadeiro brasileiro feito com chocolate amargo de alta qualidade e granulado belo."},
        {"id": 4, "titulo": "Café Gelado Cremoso", "texto": "Refrescante receita de café gelado com leite cremoso e um toque de baunilha para dias quentes."},
        {"id": 5, "titulo": "Torta de Maçã Tradicional", "texto": "Torta de maçã caseira com crosta crocante e recheio temperado com canela e açúcar mascavo."}
    ]

corpus = carregar_corpus()
textos_purificados = [doc["texto"] for doc in corpus]

# Configurando o vetorizador TF-IDF (com remoção simples de stopwords em português)
stopwords_pt = [
    "a", "o", "e", "é", "de", "do", "da", "em", "um", "para", "com", "não", "uma", 
    "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", 
    "ele", "das", "seu", "sua", "ou", "quando", "muito", "nos", "já", "está", "eu", 
    "também", "só", "pelo", "pela", "até", "isso", "she", "eles", "esta", "entre", 
    "depois", "sem", "mesmo", "aos", "suas", "quem", "nas", "me", "esse", "eles", 
    "você", "sua", "faz", "fazer"
]

vetorizador = TfidfVectorizer(stop_words=stopwords_pt, lowercase=True)
matriz_tfidf = vetorizador.fit_transform(textos_purificados)

# Título principal
st.title("🔍 Por Trás do Botão de Busca: Como o Motor Funciona?")
st.markdown("Experimente digitar uma busca abaixo (ex: *bolo de chocolate* ou *café cremoso*) e veja as etapas que acontecem em milissegundos!")

# Entrada do usuário
query = st.text_input("O que você está procurando?", "bolo de chocolate")

if query:
    st.divider()
    
    # -------------------------------------------------------------------------
    # ETAPA 1: Tratamento da Query (Tokenização e Limpeza)
    # -------------------------------------------------------------------------
    st.subheader("Etapa 1: O Tratamento da Query (Limpeza e Tokenização)")
    
    # Simula a limpeza
    query_minuscula = query.lower()
    tokens_brutos = re.findall(r'\b\w+\b', query_minuscula)
    tokens_filtrados = [t for t in tokens_brutos if t not in stopwords_pt]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Query Original", query)
    with col2:
        st.metric("Tokens Brutos", len(tokens_brutos))
        st.caption(str(tokens_brutos))
    with col3:
        st.metric("Após Remover Stopwords", len(tokens_filtrados))
        st.caption(str(tokens_filtrados))
        
    st.info("💡 **Conceito didático:** O motor remove palavras muito comuns (como 'de', 'com', 'a') que não ajudam a diferenciar o assunto principal.")

    # -------------------------------------------------------------------------
    # ETAPA 2: O Índice Invertido
    # -------------------------------------------------------------------------
    st.subheader("Etapa 2: Consultando o Índice Invertido")
    st.markdown("O índice invertido diz instantaneamente **quais documentos** contêm cada palavra-chave.")
    
    vocabulario = vetorizador.vocabulary_
    dados_indice = []
    
    for token in tokens_filtrados:
        if token in vocabulario:
            idx_termo = vocabulario[token]
            docs_encontrados = []
            for i, doc in enumerate(corpus):
                # Se o peso TF-IDF for maior que 0 naquele doc
                if matriz_tfidf[i, idx_termo] > 0:
                    docs_encontrados.append(f"Doc {doc['id']}: {doc['titulo']}")
            dados_indice.append({
                "Palavra-Chave": token,
                "Documentos Encontrados": ", ".join(docs_encontrados) if docs_encontrados else "Nenhum"
            })
        else:
            dados_indice.append({
                "Palavra-Chave": token,
                "Documentos Encontrados": "❌ Palavra não encontrada no acervo"
            })
            
    if dados_indice:
        st.table(pd.DataFrame(dados_indice))
    else:
        st.warning("Nenhuma palavra significativa restante após a limpeza.")

    # -------------------------------------------------------------------------
    # ETAPA 3: O Ranking de Relevância (TF-IDF & Similaridade)
    # -------------------------------------------------------------------------
    st.subheader("Etapa 3: Cálculo de Relevância e Ranking")
    st.markdown("O motor calcula a afinidade (pontuação matemática) de cada documento comparando os pesos das palavras.")

    # Vetoriza a query do usuário e calcula similaridade de cosseno
    vetor_query = vetorizador.transform([query])
    similaridades = cosine_similarity(vetor_query, matriz_tfidf).flatten()

    # Cria dataframe com os resultados
    resultados = []
    for i, doc in enumerate(corpus):
        resultados.append({
            "ID": doc["id"],
            "Título": doc["titulo"],
            "Trecho": doc["texto"],
            "Pontuação de Relevância": similaridades[i]
        })
        
    df_resultados = pd.DataFrame(resultados).sort_values(by="Pontuação de Relevância", ascending=False)
    
    # Exibe os resultados formatados
    for index, row in df_resultados.iterrows():
        score = row["Pontuação de Relevância"]
        cor_container = "border-left: 5px solid #4CAF50;" if score > 0 else "border-left: 5px solid #ccc;"
        
        with st.container(border=True):
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"#### 📄 {row['Título']}")
                st.write(row['Trecho'])
            with cols[1]:
                st.metric("Score (Afinidade)", f"{score:.3f}")

else:
    st.info("Digite uma busca acima para ver a mágica da Recuperação de Informação acontecer!")

# Rodapé explicativo
st.divider()
st.markdown("### 🎓 Dica para Apresentação:")
st.markdown("Você pode abrir este app em um projetor ou notebook durante a sua oficina e pedir para o público sugerir termos malucos, testando quais receitas aparecem no topo e por quê!")