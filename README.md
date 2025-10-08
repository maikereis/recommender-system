# Sistema de Recomendação de Filmes

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/)

---

Implementação educacional de dois algoritmos fundamentais de sistemas de recomendação baseada no livro "Practical Recommender Systems" de Kim Falk.

## Visão Geral

Este projeto implementa e compara duas abordagens clássicas de recomendação:

- **Filtragem Colaborativa Baseada em Itens**: Utiliza padrões de avaliação de usuários para encontrar itens similares
- **Filtragem Baseada em Conteúdo**: Utiliza metadados (gêneros, tags) para recomendar itens com características similares

## Fundamentos Teóricos

### Filtragem Colaborativa (Capítulo 8, Falk)

A implementação utiliza métodos de vizinhança baseados em itens:

**Similaridade Cosseno entre itens:**
```
sim(i, j) = (Σ r_u,i × r_u,j) / (||r_i|| × ||r_j||)
```

**Predição de rating:**
```
r̂(u,i) = Σ(sim(i,j) × r(u,j)) / Σ(sim(i,j))
```
onde j são os k-vizinhos mais próximos de i que o usuário u avaliou.

### Filtragem Baseada em Conteúdo (Capítulo 10, Falk)

Utiliza TF-IDF para vetorização de metadados textuais:

**TF-IDF:**
```
TF-IDF(t,d) = TF(t,d) × log(N / DF(t))
```

**Similaridade:**
```
sim(i,j) = cos(v_i, v_j) = (v_i · v_j) / (||v_i|| × ||v_j||)
```

## Dataset

MovieLens dataset via Kaggle:
- 610 usuários
- 9.742 filmes
- 100.836 avaliações
- 3.683 tags

**Esparsidade da matriz usuário-item: ~98%**


## Instalação

```bash
pip install -r requirements.txt

# Baixar stopwords do NLTK
python -c "import nltk; nltk.download('stopwords')"
```

## Uso

### Filtragem Colaborativa

```python
# Gerar recomendações CF para um usuário
recommendations = recommend_cf(
    user_id=1,
    user_item_values=user_item_matrix_values,
    user_ids=user_ids,
    movie_ids=movie_ids,
    item_similarity_dense=item_similarity_dense,
    top_k_neighbors=top_k_neighbors,
    movies=movies,
    top_n=10
)
```

### Filtragem Baseada em Conteúdo

```python
# Gerar recomendações baseadas em conteúdo
recommendations = recommend_content(
    user_id=1,
    ratings=ratings,
    movies_content=movies_content,
    content_similarity=content_similarity,
    top_n=10,
    min_rating=4.0
)
```

## Análise Comparativa

| Aspecto | Filtragem Colaborativa | Baseada em Conteúdo |
|---------|------------------------|---------------------|
| **Fonte de dados** | Interações usuário-item | Metadados dos itens |
| **Cold start (novos itens)** | Problemático | Funciona bem |
| **Cold start (novos usuários)** | Problemático | Problemático |
| **Serendipidade** | Alta | Baixa |
| **Explicabilidade** | Difícil | Fácil |
| **Escalabilidade** | Desafiadora | Melhor |
| **Superespecialização** | Não | Sim |

## Melhorias Futuras

### Algorítmicas
- Sistema híbrido combinando CF e content-based
- Fatoração de matriz (SVD, ALS) para redução dimensional
- Deep learning para embeddings latentes
- Incorporação de sinais temporais e contextuais

### Engenharia
- Pipeline de processamento batch vs online
- Sistema de cache para recomendações frequentes
- A/B testing framework para avaliar melhorias
- Métricas offline: RMSE, Precision@K, Recall@K, NDCG

### Features
- Diversidade nas recomendações
- Explicabilidade aumentada
- Reranking baseado em negócio
- Cold start mitigation strategies

## Referências

**Livro Principal:**
- Falk, K. (2019). Practical Recommender Systems. Manning Publications.
  - Capítulo 8: Neighborhood-Based Collaborative Filtering
  - Capítulo 10: Content-Based Filtering

**Datasets:**
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets. ACM TIIS.
