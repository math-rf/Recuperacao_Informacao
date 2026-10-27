import streamlit as st
import pandas as pd
import ir_utils
import sys
import io
import re

# Força o ecossistema de I/O a utilizar UTF-8 no Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configuração da página
st.set_page_config(
    page_title="Laboratório de RI: mMARCO-PT",
    page_icon="🔍",
    layout="wide"
)

# Função para aplicar o destaque visual (Highlight) nos termos da query
def destacar_termos(texto, tokens):
    if not tokens:
        return texto
    
    tokens_ordenados = sorted(list(set(tokens)), key=len, reverse=True)
    texto_destacado = texto
    
    for token in tokens_ordenados:
        if len(token) > 1:
            padrao = re.compile(re.escape(token), re.IGNORECASE)
            texto_destacado = padrao.sub(
                lambda m: f"<mark style='background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 4px; font-weight: bold;'>{m.group(0)}</mark>", 
                texto_destacado
            )
            
    return texto_destacado

# Carregamento do Corpus em cache
@st.cache_data
def carregar_dados_corpus():
    return ir_utils.carregar_corpus_mmarco(limite_docs=1500)

with st.spinner("Carregando o corpus mMARCO da literatura... Aguarde um instante."):
    corpus = carregar_dados_corpus()

corpus_textos = [doc["texto"] for doc in corpus]

# Título principal e descrição
st.title("🔍 Laboratório Interativo de Recuperação de Informação")
st.markdown(f"Explorando **{len(corpus)} documentos reais** de forma divertida e interativa.")

# Criação das Abas Principais
aba_laboratorio, aba_didatica = st.tabs(["🚀 Laboratório Prático de Busca", "🎈 Como os Modelos Pensam?"])

# =============================================================================
# ABA 1: O LABORATÓRIO PRÁTICO
# =============================================================================
with aba_laboratorio:
    # Sidebar de Configurações Didáticas
    st.sidebar.header("⚙️ Configurações do Laboratório")

    modo_exibicao = st.sidebar.radio(
        "Modo de Visualização:",
        ["Análise Detalhada (1 Modelo)", "Comparação Lado a Lado (2 Modelos)"]
    )

    modelo_selecionado = st.sidebar.selectbox(
        "Escolha o Modelo Principal:",
        [
            "Vetorial (TF-IDF + Cosseno)",
            "BM25 (Probabilístico)",
            "Similaridade de Jaccard (Conjuntos)"
        ]
    )

    k1, b = 1.5, 0.75
    if "BM25" in modelo_selecionado:
        st.sidebar.subheader("🎛️ Parâmetros do BM25")
        k1 = st.sidebar.slider("Parâmetro k1 (Saturação de TF)", 1.0, 3.0, 1.5, 0.1)
        b = st.sidebar.slider("Parâmetro b (Penalização de Comprimento)", 0.0, 1.0, 0.75, 0.1)

    # Entrada da Consulta
    query = st.text_input("Digite sua consulta de busca:", "qual a capital do brasil")

    if query:
        st.divider()
        
        # ETAPA 1: Tratamento da Query
        st.subheader("Etapa 1: O Tratamento da Query (Limpeza e Tokenização)")
        tokens_brutos = ir_utils.preprocessar_texto(query, remover_stopwords=False)
        tokens_filtrados = ir_utils.preprocessar_texto(query, remover_stopwords=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Query Original", f'"{query}"')
        with col2:
            st.metric("Tokens Brutos", len(tokens_brutos))
            st.caption(str(tokens_brutos))
        with col3:
            st.metric("Após Remover Stopwords", len(tokens_filtrados))
            st.caption(str(tokens_filtrados))

        # ETAPA 2: Análise de Relevância e Pesos dos Termos
        st.subheader("Etapa 2: Análise de Relevância dos Termos da Consulta")
        try:
            _, vectorizer_analise, matriz_analise = ir_utils.modelo_vetorial_tfidf(corpus_textos, query)
            vocab = vectorizer_analise.vocabulary_
            idf_values = vectorizer_analise.idf_

            dados_pesos = []
            for token in tokens_filtrados:
                if token in vocab:
                    idx_termo = vocab[token]
                    peso_idf = idf_values[idx_termo]
                    docs_contendo = (matriz_analise[:, idx_termo].toarray() > 0).sum()
                    dados_pesos.append({
                        "Termo": token,
                        "Peso IDF (Raridade Global)": round(peso_idf, 4),
                        "Documentos na Amostra que Contêm": int(docs_contendo),
                        "Nível de Relevância / Discriminatório": "🔥 Alto (Termo Raro/Informativo)" if peso_idf > 3.0 else "⚡ Moderado/Comum"
                    })
                else:
                    dados_pesos.append({
                        "Termo": token,
                        "Peso IDF (Raridade Global)": 0.0,
                        "Documentos na Amostra que Contêm": 0,
                        "Nível de Relevância / Discriminatório": "❌ Ausente no Acervo"
                    })

            if dados_pesos:
                st.table(pd.DataFrame(dados_pesos))
        except Exception as e:
            st.warning(f"Não foi possível gerar a tabela detalhada de pesos: {e}")

        # ETAPA 3: Índice Invertido
        st.subheader("Etapa 3: Consultando o Índice Invertido")
        dados_indice = []
        for token in tokens_filtrados:
            docs_encontrados = []
            for doc in corpus:
                doc_tokens = ir_utils.preprocessar_texto(doc["texto"], remover_stopwords=True)
                if token in doc_tokens:
                    docs_encontrados.append(doc['id'])
                    if len(docs_encontrados) >= 5:
                        break
                    
            dados_indice.append({
                "Palavra-Chave": token,
                "Total de Docs Encontrados (DF)": len(docs_encontrados),
                "Exemplos de IDs": ", ".join([f"{d[:6]}..." for d in docs_encontrados]) if docs_encontrados else "❌ Nenhuma ocorrência"
            })
                
        if dados_indice:
            st.table(pd.DataFrame(dados_indice))

        # ETAPA 4: Ranking e Exibição dos Resultados com Destaque Visual
        if modo_exibicao == "Análise Detalhada (1 Modelo)":
            st.subheader(f"Etapa 4: Ranking de Relevância Final ({modelo_selecionado})")
            
            if "BM25" in modelo_selecionado:
                ranking = ir_utils.modelo_bm25(corpus_textos, query, k1=k1, b=b)
            elif "Vetorial" in modelo_selecionado:
                ranking, _, _ = ir_utils.executar_modelo_busca("Vetorial (TF-IDF + Cosseno)", corpus_textos, query)
            else:
                ranking, _, _ = ir_utils.executar_modelo_busca("Similaridade de Jaccard (Conjuntos)", corpus_textos, query)

            for idx, score in ranking[:5]:
                doc = corpus[idx]
                texto_doc = doc['texto']
                
                resumo_cru = texto_doc[:200] + ("..." if len(texto_doc) > 200 else "")
                resumo_destacado = destacar_termos(resumo_cru, tokens_filtrados)

                with st.container(border=True):
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.markdown(f"#### 📄 {doc['titulo']}")
                        st.markdown(resumo_destacado, unsafe_allow_html=True)
                        
                        if len(texto_doc) > 200:
                            with st.expander("📖 Ver texto completo do documento (com destaque)"):
                                texto_completo_destacado = destacar_termos(texto_doc, tokens_filtrados)
                                st.markdown(texto_completo_destacado, unsafe_allow_html=True)
                                
                        st.caption(f"ID do Corpus: `{doc['id']}`")
                    with cols[1]:
                        st.metric("Score", f"{score:.4f}")

        else:
            st.subheader("Etapa 4: Comparação de Modelos Lado a Lado")
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.markdown("### 📊 BM25 (Probabilístico)")
                ranking_bm25 = ir_utils.modelo_bm25(corpus_textos, query, k1=k1, b=b)
                for idx, score in ranking_bm25[:3]:
                    doc = corpus[idx]
                    texto_doc = doc['texto']
                    resumo = destacar_termos(texto_doc[:150] + ("..." if len(texto_doc) > 150 else ""), tokens_filtrados)

                    with st.container(border=True):
                        st.markdown(f"**[{doc['id'][:6]}...]** — Score: `{score:.3f}`")
                        st.markdown(resumo, unsafe_allow_html=True)
                        with st.expander("📖 Ler documento na íntegra"):
                            st.markdown(destacar_termos(texto_doc, tokens_filtrados), unsafe_allow_html=True)

            with col_m2:
                st.markdown("### 📊 Vetorial (TF-IDF)")
                ranking_tfidf, _, _ = ir_utils.executar_modelo_busca("Vetorial (TF-IDF + Cosseno)", corpus_textos, query)
                for idx, score in ranking_tfidf[:3]:
                    doc = corpus[idx]
                    texto_doc = doc['texto']
                    resumo = destacar_termos(texto_doc[:150] + ("..." if len(texto_doc) > 150 else ""), tokens_filtrados)

                    with st.container(border=True):
                        st.markdown(f"**[{doc['id'][:6]}...]** — Score: `{score:.3f}`")
                        st.markdown(resumo, unsafe_allow_html=True)
                        with st.expander("📖 Ler documento na íntegra"):
                            st.markdown(destacar_termos(texto_doc, tokens_filtrados), unsafe_allow_html=True)

    else:
        st.info("Digite uma pergunta ou termo de busca acima para iniciar a exploração!")


# =============================================================================
# ABA 2: EXPLICAÇÃO LÚDICA PARA CRIANÇAS E JOVENS
# =============================================================================
with aba_didatica:
    st.header("🎈 Como o Computador Encontra o que Você Procura?")
    st.markdown("Imagine que o computador é um **bibliotecário super rápido e inteligente**, mas que não entende as coisas do jeito que a gente entende. Ele precisa de regras matemáticas mágicas para adivinhar qual livro ou história você quer ler!")

    st.divider()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 🧩 1. O Detetive de Peças (Jaccard)")
        st.markdown("""
        * **Como funciona na história:** É como brincar de montar um quebra-cabeça ou comparar duas mochilas. O detetive olha as palavras da sua pergunta e vê quantas delas batem exatamente com as palavras da história.
        * **O lema dele:** *"Se tem as mesmas palavras, é amigo!"*
        * **Para quem é fácil entender:** Lembra quando você confere quais figurinhas repetidas você tem iguais às do seu amigo.
        """)

    with col_b:
        st.markdown("### ⚖️ 2. O Inspetor de Raras e Comuns (TF-IDF)")
        st.markdown("""
        * **Como funciona na história:** Esse inspetor é muito esperto. Ele sabe que a palavra **"o"** ou **"de"** aparece em todo lugar e não ajuda em quase nada. Mas se você falar uma palavra rara (como *"Dinossauro"* ou *"Astronauta"*), ele dá **muito mais pontos** para a história que tem essa palavra mágica!
        * **O lema dele:** *"Quanto mais rara e importante a palavra, mais valiosa ela é!"*
        """)

    with col_c:
        st.markdown("### 🏆 3. O Campeão dos Campeões (BM25)")
        st.markdown("""
        * **Como funciona na história:** É o modelo mais poderoso de todos (usado pelos grandes sites de busca do mundo!). Ele percebe quando uma palavra se repete exageradamente só para tentar enganar o sistema, e também não deixa textos gigantes passarem na frente só por serem longos.
        * **O lema dele:** *"Justiça máxima: ganha quem realmente responde à sua dúvida com qualidade!"*
        """)

    st.divider()
    st.info("💡 **Dica de Ouro:** Volte na aba **'Laboratório Prático'**, mude os modelos de busca e veja como o pódio das histórias muda de lugar dependendo de qual 'detetive' está trabalhando na busca!")