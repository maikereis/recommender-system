# %% [markdown]
# # Sistema de Recomendação de Filmes
# 
# Este caderno implementa duas abordagens fundamentais de sistemas de recomendação baseadas em
# "Sistemas de Recomendação Práticos" de Kim Falk:
# 
# 1. **Filtragem Colaborativa**: Utiliza interações usuário-item para encontrar usuários ou itens semelhantes
# 2. **Filtragem Baseada em Conteúdo**: Utiliza características do item (gêneros, tags) para recomendar itens semelhantes
# 
# ## Contexto Teórico
# 
# ### Filtragem Colaborativa na Vizinhança (Capítulo 8, Falk)
# - **CF Baseada no Usuário**: Encontra usuários semelhantes ao usuário-alvo e recomenda os itens que eles gostaram
# - **CF Baseada no Item**: Encontra itens semelhantes aos que o usuário gostou (nós implementamos isso)
# - **Métodos de Vizinhança**: Utiliza k-vizinhos mais próximos com base em métricas de similaridade
# - **Métrica de Similaridade**: A similaridade do cosseno mede o ângulo entre os vetores de classificação
# 
# ### Filtragem Baseada em Conteúdo (Capítulo 10, Falk)
# - Utiliza metadados de itens (gêneros, tags) para calcular a similaridade dos itens
# - **TF-IDF**: Frequência de Termos - Frequência Inversa de Documentos pondera os termos por importância em todo o corpus
# - **Modelo de Espaço Vetorial**: Representa itens como vetores no espaço de características
# - Recomenda itens semelhantes àqueles que o usuário avaliou positivamente
# 

# %%
import re
import warnings
import unicodedata
import kagglehub
import polars as pl
import numpy as np
from pathlib import Path
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Carregamento dos Dados
# 
# Utilizamos o dataset MovieLens do Kaggle, que contém avaliações de filmes, metadados e tags geradas por usuários.
# O dataset é baixado automaticamente via `kagglehub`.

# %%
# Download latest version
path = kagglehub.dataset_download("aigamer/movie-lens-dataset")

# tmdb_path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")

# %% [markdown]
# ### Funções Auxiliares
# 
# Definimos funções utilitárias para:
# - **Calcular esparsidade**: Mede a proporção de valores zero na matriz
# - **Remover acentos**: Normaliza texto para processamento
# - **Limpar texto**: Remove stopwords e normaliza para análise de conteúdo

# %%
STOPWORDS = set(stopwords.words("english"))


def calculate_sparsity(df: pl.DataFrame) -> float:
    """Calcula a proporção de valores zero/nulos no DataFrame."""
    total_elements = df.shape[0] * df.shape[1]

    zeros = (
        df.fill_null(0)
        .select(pl.sum_horizontal(df.fill_null(0) == 0))
        .to_series()
        .sum()
    )

    sparsity = zeros / total_elements
    return sparsity


def remove_accents(text: str) -> str:
    """
    Remove acentuação de um texto usando normalização Unicode.
    Útil para padronizar termos de diferentes idiomas.
    """
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def clean_text(text):
    """Remove acentos, converte para minúsculas e filtra stopwords."""
    text = remove_accents(text.lower())
    tokens = text.split()
    filtered = " ".join([t for t in tokens if t not in STOPWORDS])
    return filtered

# %% [markdown]
# ## Carregamento e Exploração dos Dados

# %%
# Load datasets
links = pl.read_csv(Path(path, "links.csv"))
movies = pl.read_csv(Path(path, "movies.csv"))
ratings = pl.read_csv(Path(path, "ratings.csv"))
tags = pl.read_csv(Path(path, "tags.csv"))

print(f"Movies: {len(movies)}, Ratings: {len(ratings)}, Tags: {len(tags)}")

# %% [markdown]
# ## Dicionário de Dados
# 
# * `userId`: Identificador único de cada usuário
# * `movieId`: Identificador único de cada filme
# * `title`: Título do filme
# * `rating`: notas de 0 a 5
# * `genres`: Gênero de cada filme com os seguintes valores possíveis;
#     * Action
#     * Adventure
#     * Animation
#     * Children's
#     * Comedy
#     * Crime
#     * Documentary
#     * Drama
#     * Fantasy
#     * Film-Noir
#     * Horror
#     * Musical
#     * Mystery
#     * Romance
#     * Sci-Fi
#     * Thriller
#     * War
#     * Western
#     * (no genres listed)
# * `tag`: Metadados gerados pelos usuários
# * `timestamp`: Unix timestamp da ação

# %%
print("Filmes:")
display(movies.head())
print("\nAvaliações:")
display(ratings.head())
print("\nTags:")
display(tags.head())

# %% [markdown]
# ## Pré-processamento
# 
# Preparamos os dados para ambas as abordagens:
# 1. **Normalizar gêneros**: Substituir separadores `|` por espaços para facilitar vetorização TF-IDF
# 2. **Converter timestamps**: Transformar Unix timestamps em objetos datetime legíveis
# 3. **Criar matriz usuário-item**: Estrutura fundamental para filtragem colaborativa

# %%
movies = movies.with_columns(
    pl.col("genres")
    .str.replace_all(r"\|", " ")  # replace '|' with space
    .alias("genres")
)
ratings = ratings.with_columns(pl.from_epoch("timestamp", time_unit="s"))
tags = tags.with_columns(pl.from_epoch("timestamp", time_unit="s"))

# %%
tags

# %%
print("Pré-processamento completo!")
print("Filmes:")
display(movies.head())
print("\nAvaliações:")
display(ratings.head())
print("\nTags:")
display(tags.head())

# %% [markdown]
# ## Implementação da Filtragem Colaborativa
# 
# ### Fundamentação Teórica (Capítulo 8, Falk)
# 
# A filtragem colaborativa pressupõe que os usuários que concordaram no passado concordarão no futuro.
# O Capítulo 8 aborda **métodos baseados em vizinhança**, que encontram usuários ou itens semelhantes.
# 
# Implementamos a **filtragem colaborativa baseada em itens** que:
# 
# 1. Calcula a similaridade item-item usando a similaridade de cosseno na matriz usuário-item
# 2. Para um usuário-alvo, identifica os itens que ele avaliou positivamente
# 3. Encontra itens semelhantes que o usuário ainda não viu (a "vizinhança")
# 4. Classifica as recomendações pela avaliação prevista
# 
# **Fórmula de Similaridade de Cosseno**:
# $\text{sim}(i, j) = \frac{\sum_{u \in U} r_{u,i} \cdot r_{u,j}}{\sqrt{\sum_{u \in U} r_{u,i}^2} \cdot \sqrt{\sum_{u \in U} r_{u,j}^2}}$
# 
# Onde $r_{u,i}$ é a avaliação do usuário $u$ para o item $i$.

# %% [markdown]
# ### Construção da Matriz Usuário-Item
# 
# A matriz usuário-item é o coração da filtragem colaborativa. Cada célula $(u, i)$ contém a avaliação
# do usuário $u$ para o item $i$. Usamos pivotamento para transformar os dados em formato longo
# para uma matriz 2D onde:
# - **Linhas**: representam usuários
# - **Colunas**: representam filmes
# - **Valores**: avaliações (ratings)

# %%
user_item_matrix = ratings.pivot(
    values="rating",
    index="userId",
    columns="movieId",
    aggregate_function="max",
)

# Store userIs and movieIds
user_ids = user_item_matrix.select("userId").to_series().to_numpy()
movie_ids = [col for col in user_item_matrix.columns if col != "userId"]

# %%
user_item_matrix.head()

# %% [markdown]
# ### Análise de Esparsidade
# 
# Nossa matriz usuário-item tem dimensões (611 × 9,725) e consome 45MB. 
# Este tamanho é gerenciável, mas em sistemas reais com milhões de usuários e itens,
# a representação densa seria inviável.

# %%
# Estimated memory usage in bytes
memory_bytes = user_item_matrix.estimated_size()
print(f"Estimated memory usage: {memory_bytes / (1024**2):.2f} MB")

# %% [markdown]
# A matriz apresenta ~98% de esparsidade, típico em sistemas de recomendação onde
# a maioria dos usuários interage com apenas uma pequena fração dos itens disponíveis.
# Isso justifica o uso de representações esparsas para eficiência computacional e de memória.

# %%
sparsity = calculate_sparsity(user_item_matrix)
print(f"Sparsity: {sparsity:.2%}")

# %% [markdown]
# ### Distribuição de Avaliações por Usuário
# 
# Analisamos quantos filmes cada usuário avaliou para entender o padrão de engajamento.
# A mediana e moda revelam que a maioria dos usuários avalia poucos filmes em relação
# ao catálogo total de 9.742 filmes disponíveis.

# %%
ratings_per_user = ratings.group_by("userId").agg(
    pl.count("movieId").alias("num_ratings")
)

median_ratings = ratings_per_user.select(pl.col("num_ratings").median()).item()
mode_ratings = ratings_per_user.select(pl.col("num_ratings").mode()).item()

print(
    f"Até 50% dos usuários avaliam {median_ratings:.0f} filmes ou menos.\n"
    f"O número mais comum de avaliações por usuário é {mode_ratings}."
)

# %% [markdown]
# ### Conversão para Matriz Esparsa
# 
# Convertemos para formato CSR (Compressed Sparse Row) do SciPy, que:
# - **Armazena apenas valores não-zero** e suas posições
# - **Reduz drasticamente o uso de memória** (de 45MB para ~1MB neste caso)
# - **Acelera operações matriciais** quando a matriz é esparsa
# - É a estrutura ideal para cálculo de similaridade em matrizes esparsas

# %%
from scipy.sparse import csr_matrix

user_item_matrix_values = user_item_matrix.select(movie_ids).fill_null(0).to_numpy()
user_item_matrix_crs = csr_matrix(user_item_matrix_values.T)

# %%
user_item_matrix_crs

# %% [markdown]
# ### Cálculo da Similaridade Item-Item
# 
# Computamos a matriz de similaridade cosseno entre todos os pares de filmes.
# Cada célula $(i, j)$ representa quão similares são os filmes $i$ e $j$ baseado
# nos padrões de avaliação dos usuários.
# 
# **Intuição**: Filmes com vetores de avaliação similares (avaliados de forma parecida
# pelos mesmos usuários) terão alta similaridade cosseno.
# 
# $$
# \text{similarity}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \, \|\mathbf{v}_j\|}
# $$
# 
# onde $\mathbf{v}_i$ e $\mathbf{v}_j$ são os vetores de avaliação dos itens $i$ e $j$.

# %%
item_similarity = cosine_similarity(user_item_matrix_crs, dense_output=False)

# %% [markdown]
# ### Pré-computação dos K-Vizinhos Mais Próximos
# 
# Para eficiência em tempo de execução, pré-computamos os top-K vizinhos mais similares
# para cada filme. Isso evita percorrer toda a matriz de similaridade durante a predição,
# reduzindo a complexidade de O(n) para O(k) onde k << n.

# %%
item_similarity_dense = item_similarity.toarray()

# Precompute top-K neighbors for each item
k_neighbors = 20
top_k_neighbors = {}

for i, movie_id in enumerate(movie_ids):
    # Get similarities for this movie
    sims = item_similarity_dense[i]
    # Get indices of top-k items (excluding self)
    top_indices = np.argsort(sims)[-k_neighbors:][::-1]
    top_k_neighbors[movie_id] = [movie_ids[idx] for idx in top_indices]

# %% [markdown]
# ### Funções de Predição e Recomendação
# 
# **Predição de Rating**: Estima a avaliação de um usuário para um filme não-visto
# usando média ponderada das avaliações dos filmes similares que o usuário já avaliou.
# 
# **Algoritmo de Recomendação**:
# 1. Identifica filmes não-vistos pelo usuário
# 2. Prediz rating para cada filme não-visto
# 3. Ordena por rating predito decrescente
# 4. Retorna top-N recomendações

# %%
def predict_rating(
    user_id,
    movie_id,
    user_item_values,
    user_ids,
    movie_ids,
    item_similarity_dense,
    top_k_neighbors,
):
    """
    Prediz o rating que um usuário daria a um filme usando CF baseada em itens.
    
    Fórmula: r̂(u,i) = Σ(sim(i,j) × r(u,j)) / Σ(sim(i,j))
    onde j são os vizinhos de i que u avaliou.
    """
    # Find user index
    user_idx = np.where(user_ids == user_id)[0]
    if len(user_idx) == 0 or movie_id not in movie_ids:
        return 0.0

    user_idx = user_idx[0]
    movie_idx = movie_ids.index(movie_id)

    # Get user's ratings
    user_ratings = user_item_values[user_idx]

    # Get neighbors that user has rated
    neighbors = top_k_neighbors[movie_id]
    neighbor_indices = [
        i for i, mid in enumerate(movie_ids) if mid in neighbors and user_ratings[i] > 0
    ]

    if not neighbor_indices:
        return 0.0

    # Get similarities and ratings for neighbors
    similarities = item_similarity_dense[movie_idx, neighbor_indices]
    neighbor_ratings = user_ratings[neighbor_indices]

    weighted_sum = np.sum(similarities * neighbor_ratings)
    similarity_sum = np.sum(similarities)

    return float(weighted_sum / similarity_sum) if similarity_sum > 0 else 0.0


def recommend_cf(
    user_id,
    user_item_values,
    user_ids,
    movie_ids,
    item_similarity_dense,
    top_k_neighbors,
    movies,
    top_n=10,
):
    """Gera top-N recomendações de filmes para um usuário usando CF baseada em itens."""
    # Find user index
    user_idx = np.where(user_ids == user_id)[0]
    if len(user_idx) == 0:
        return pl.DataFrame()

    user_idx = user_idx[0]
    user_ratings = user_item_values[user_idx]

    # Find unrated movies
    unrated_indices = np.where(user_ratings == 0)[0]
    unrated_movie_ids = [movie_ids[i] for i in unrated_indices]

    predictions = []
    for movie_id in unrated_movie_ids:
        pred = predict_rating(
            user_id,
            movie_id,
            user_item_values,
            user_ids,
            movie_ids,
            item_similarity_dense,
            top_k_neighbors,
        )
        if pred > 0:
            predictions.append({"movieId": movie_id, "predicted_rating": pred})

    if not predictions:
        return pl.DataFrame()

    # Create Polars DataFrame and sort
    recommendations = (
        pl.DataFrame(predictions)
        .with_columns(pl.col("movieId").cast(pl.Int64))
        .sort("predicted_rating", descending=True)
        .head(top_n)
    )

    # Join with movies data
    return recommendations.join(
        movies.select(["movieId", "title", "genres"]), on="movieId", how="left"
    )

# %% [markdown]
# ### Teste do Sistema de Filtragem Colaborativa
# 
# Testamos o recomendador CF com um usuário específico:
# 1. Mostramos os filmes que o usuário avaliou com maiores notas
# 2. Geramos recomendações baseadas em similaridade com esses filmes
# 3. A ideia é que as recomendações sejam similares aos filmes bem avaliados

# %%
test_user = 1

# %%
# User's top-rated movies
user_top_ratings = (
    ratings.filter(pl.col("userId") == test_user)
    .join(movies.select(["movieId", "title", "genres"]), on="movieId", how="left")
    .sort("rating", descending=True)
    .head(5)
    .select(["title", "rating", "genres"])
)
print("\nUser's Top-Rated Movies:")
print(user_top_ratings)

# %%
cf_recommendations = recommend_cf(
    test_user,
    user_item_matrix_values,
    user_ids,
    movie_ids,
    item_similarity_dense,
    top_k_neighbors,
    movies,
    10,
)
print("Collaborative Filtering Recommendations:")
print(cf_recommendations)

# %% [markdown]
# ## Implementação de Filtragem Baseada em Conteúdo
# 
# ### Fundamentação Teórica (Capítulo 10, Falk)
# 
# A filtragem baseada em conteúdo recomenda itens semelhantes aos que o usuário gostou,
# com base nas características dos itens. O Capítulo 10 aborda a abordagem completa baseada em conteúdo:
# 
# 1. **Extração de Características**: Extrai características dos metadados do item (gêneros, tags, descrições)
# 2. **TF-IDF (Frequência de Termos - Frequência Inversa do Documento)**: Pondera os termos por importância
# - **TF**: Com que frequência um termo aparece em um documento (importância local)
# - **IDF**: Quão raro o termo é em todos os documentos (importância global)
# - Fórmula: $\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}$
# 
# 3. **Modelo de Espaço Vetorial**: Cada item é um vetor em um espaço de características de alta dimensão
# 4. **Similaridade de Cossenos**: Mede a similaridade entre vetores de características
# 
# **Vantagens** (Capítulo 10):
# - Sem problemas de inicialização a frio para itens com metadados
# - Recomendações transparentes (explicáveis ​​por características)
# - Independência do usuário (não precisa de outros usuários dados)
# 
# **Desafios** (Capítulo 10):
# - Serendipidade limitada (superespecialização)
# - Requer metadados de itens ricos e de alta qualidade
# - A engenharia de recursos é específica do domínio

# %% [markdown]
# ### Agregação de Tags por Filme
# 
# Múltiplos usuários podem atribuir múltiplas tags ao mesmo filme, gerando várias linhas
# por filme no dataset de tags. Agregamos todas as tags de cada filme em uma única string
# para facilitar a vetorização TF-IDF.

# %%
movie_tags = tags.group_by("movieId").agg(
    pl.col("tag").str.to_lowercase().str.join(" ").alias("tags")
)

# %%
movie_tags.head()

# %% [markdown]
# ### Construção do Dataset de Conteúdo
# 
# Combinamos todas as features de conteúdo (gêneros + tags) em um único campo `content`
# que representa o "documento" de cada filme. Este documento será vetorizado para
# calcular similaridades baseadas em conteúdo.

# %%
movies_content = movies.join(movie_tags, on="movieId", how="left")
movies_content = movies_content.with_columns(pl.col("tags").fill_null(""))
movies_content.head()

# %%
movies_content = movies_content.with_columns(
    pl.concat_str([pl.col("genres"), pl.col("tags")]).alias("content")
)

movies_content = movies_content.sort("movieId")

# %% [markdown]
# ### Limpeza e Normalização de Texto
# 
# Aplicamos as funções de limpeza definidas anteriormente para:
# - Remover acentos
# - Converter para minúsculas
# - Filtrar stopwords
# 
# Isso melhora a qualidade da vetorização TF-IDF ao focar em termos informativos.

# %%
movies_content = movies_content.with_columns(
    pl.col("content").map_elements(clean_text, return_dtype=pl.String).alias("content")
)

# %%
movies_content.head()

# %% [markdown]
# ### Vetorização TF-IDF
# 
# Transformamos os documentos de texto em vetores numéricos usando TF-IDF:
# - **Unigramas e bigramas** (n-grams 1-2): Captura termos individuais e pares de termos consecutivos
# - **Stopwords removidas**: Foco em termos semanticamente relevantes
# - **Matriz esparsa resultante**: Cada filme é um vetor de pesos TF-IDF

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=None,  # Limit features for efficiency
    stop_words="english",  # Remove common English words
    ngram_range=(1, 2),  # Use unigrams and bigrams
)
tfidf_matrix = tfidf.fit_transform(movies_content["content"])

# %% [markdown]
# ### Cálculo da Similaridade de Conteúdo
# 
# Computamos similaridade cosseno entre os vetores TF-IDF de todos os filmes.
# Filmes com vocabulário similar (gêneros e tags parecidas) terão alta similaridade,
# independentemente das avaliações dos usuários.

# %%
content_similarity = cosine_similarity(tfidf_matrix)

# %% [markdown]
# ### Funções de Busca e Recomendação Baseada em Conteúdo
# 
# **get_similar_movies**: Encontra filmes similares a um filme específico baseado apenas em conteúdo.
# 
# **recommend_content**: Gera recomendações para um usuário:
# 1. Identifica filmes bem avaliados pelo usuário (rating ≥ threshold)
# 2. Para cada filme bem avaliado, encontra filmes similares por conteúdo
# 3. Agrega scores de similaridade ponderados pelo rating do usuário
# 4. Filtra filmes já vistos
# 5. Retorna top-N por score agregado

# %%
def get_similar_movies(movie_id, movies_content, content_similarity, top_n=10):
    """
    Encontra filmes mais similares a um dado filme baseado em conteúdo.
    
    Útil para explorar a qualidade da matriz de similaridade de conteúdo
    e entender quais features (gêneros/tags) estão dirigindo as recomendações.
    """
    # Find the index of the movie
    movie_indices = np.where(movies_content["movieId"].to_numpy() == movie_id)[0]

    if len(movie_indices) == 0:
        return pl.DataFrame()

    movie_idx = movie_indices[0]

    # Get similarity scores for this movie
    similarity_scores = content_similarity[movie_idx]

    # Get indices of most similar movies (excluding self)
    similar_indices = np.argsort(similarity_scores)[::-1][1 : top_n + 1]

    # Create DataFrame with results
    similar_movies = (
        movies_content[similar_indices]
        .select(["movieId", "title", "genres"])
        .with_columns(pl.Series("similarity", similarity_scores[similar_indices]))
    )

    return similar_movies


def recommend_content(
    user_id, ratings, movies_content, content_similarity, top_n=10, min_rating=4.0
):
    """
    Gera top-n recomendações usando filtragem baseada em conteúdo.

    Algoritmo do Capítulo 10 (Content-Based Filtering):
    1. Constrói perfil do usuário a partir de itens bem avaliados (>= min_rating)
    2. Para cada item no perfil, encontra itens similares usando features de conteúdo
    3. Agrega scores de similaridade (ponderados pelo rating do usuário)
    4. Filtra itens já vistos
    5. Retorna top-n por similaridade de conteúdo agregada
    """
    # Get user's highly rated movies
    liked_movies = ratings.filter(
        (pl.col("userId") == user_id) & (pl.col("rating") >= min_rating)
    )

    if liked_movies.height == 0:
        print(f"User {user_id} has no highly rated movies (>= {min_rating})")
        return pl.DataFrame()

    print(f"User {user_id} has {liked_movies.height} highly rated movies")
    print(f"Finding content-based recommendations...")

    # Get all movies user has seen
    user_rated = ratings.filter(pl.col("userId") == user_id)
    seen_movie_ids = set(user_rated["movieId"].to_list())

    # Build aggregated similarity scores
    movie_ids_array = movies_content["movieId"].to_numpy()
    aggregated_scores = {}

    for row in liked_movies.iter_rows(named=True):
        movie_id = row["movieId"]
        user_rating = row["rating"]

        # Find index in content similarity matrix
        movie_indices = np.where(movie_ids_array == movie_id)[0]

        if len(movie_indices) == 0:
            continue

        movie_idx = movie_indices[0]

        # Get similarities and weight by user's rating
        similarities = content_similarity[movie_idx]
        weight = user_rating / 5.0

        # Accumulate weighted scores for all movies
        for i, sim_score in enumerate(similarities):
            candidate_movie_id = movie_ids_array[i]

            # Skip the movie itself
            if candidate_movie_id == movie_id:
                continue

            weighted_score = sim_score * weight

            if candidate_movie_id in aggregated_scores:
                aggregated_scores[candidate_movie_id] += weighted_score
            else:
                aggregated_scores[candidate_movie_id] = weighted_score

    # Filter out seen movies
    recommendations = {
        movie_id: score
        for movie_id, score in aggregated_scores.items()
        if movie_id not in seen_movie_ids
    }

    if len(recommendations) == 0:
        print("No new recommendations could be generated")
        return pl.DataFrame()

    # Convert to DataFrame
    recommendations_df = pl.DataFrame(
        {
            "movieId": list(recommendations.keys()),
            "content_score": list(recommendations.values()),
        }
    )

    # Sort and get top-n
    recommendations_df = recommendations_df.sort("content_score", descending=True).head(
        top_n
    )

    # Join with movie details
    result = recommendations_df.join(
        movies_content.select(["movieId", "title", "genres"]), on="movieId", how="left"
    ).select(["movieId", "title", "content_score", "genres"])

    return result

# %% [markdown]
# ### Teste de Similaridade de Conteúdo: Toy Story
# 
# Demonstramos a busca de filmes similares usando apenas características de conteúdo.
# Esperamos encontrar filmes com gêneros e tags similares a Toy Story (Animation, Children's, Comedy).

# %%
# Example: Find movies similar to a specific movie
toy_story_id = (
    movies.filter(pl.col("title").str.contains("Toy Story"))
    .select("movieId")
    .head(1)
    .item()
)
print(f"\nMovies similar to Toy Story (movieId={toy_story_id}):")
similar_to_toy_story = get_similar_movies(
    toy_story_id, movies_content, content_similarity, top_n=5
)
print(similar_to_toy_story)

# %% [markdown]
# ### Teste do Sistema de Recomendação Baseada em Conteúdo
# 
# Geramos recomendações para o mesmo usuário de teste, mas agora usando apenas
# características de conteúdo. Comparando com CF, podemos observar:
# - **CF**: Recomenda o que usuários similares gostaram (descoberta social)
# - **Content-Based**: Recomenda filmes similares aos que o usuário já gostou (consistência temática)

# %%
# Test the content-based recommender
print("\nContent-Based Recommendations:")
content_recommendations = recommend_content(
    test_user, ratings, movies_content, content_similarity, top_n=10
)
print(content_recommendations)

# %% [markdown]
# ## Conclusão e Próximos Passos
# 
# ### Comparação das Abordagens
# 
# | Aspecto | Filtragem Colaborativa | Filtragem Baseada em Conteúdo |
# |---------|------------------------|-------------------------------|
# | **Fonte de dados** | Interações usuário-item | Metadados dos itens |
# | **Cold start (novos itens)** | Problemático | Funciona bem com metadados |
# | **Cold start (novos usuários)** | Problemático | Problemático |
# | **Serendipidade** | Alta (descobre padrões inesperados) | Baixa (superespecialização) |
# | **Explicabilidade** | Difícil ("usuários como você gostaram") | Fácil ("similar a X que você gostou") |
# | **Escalabilidade** | Desafiadora (matriz cresce rapidamente) | Mais escalável (depende de features) |
# 
# ### Melhorias Possíveis
# 
# 1. **Sistema Híbrido**: Combinar CF e content-based para aproveitar vantagens de ambos
# 2. **Fatoração de Matriz**: Usar SVD ou ALS para redução de dimensionalidade e melhor generalização
# 3. **Deep Learning**: Embeddings neurais para capturar relações complexas
# 4. **Features Adicionais**: Incorporar dados temporais, contextuais, ou de texto livre (reviews)
# 5. **Avaliação Offline**: Implementar métricas como RMSE, precision@k, recall@k, NDCG
# 6. **Diversidade**: Balancear accuracy com diversidade nas recomendações
# 
# ### Referências Teóricas Aplicadas
# 
# - **Capítulo 8 (Falk)**: Neighborhood-based collaborative filtering implementada com item-based CF
# - **Capítulo 10 (Falk)**: Content-based filtering implementada com TF-IDF e cosine similarity
# - **Sparse matrices**: Uso de CSR para eficiência computacional com dados esparsos
# - **K-nearest neighbors**: Pré-computação de vizinhos para otimização de queries