# AI Module

The `ai` module contains all the artificial intelligence and machine learning components of the Linkwarden Enhancer. This module is responsible for tasks suchs as content analysis, similarity detection, tag prediction, and clustering.

## Modules

### Clustering Engine (`clustering_engine.py`)

The `ClusteringEngine` is responsible for grouping similar bookmarks together into clusters. This helps in organizing large collections of bookmarks into meaningful groups.

**Classes:**

- **`BookmarkCluster`**: A data class that represents a cluster of bookmarks, including the bookmark indices, centroid keywords, suggested collection name, and other metadata.
- **`DomainCluster`**: A data class that represents a cluster based on domain patterns.
- **`ClusteringResult`**: A data class that encapsulates the results of a clustering operation, including the list of clusters, optimal cluster count, and silhouette score.
- **`ClusteringEngine`**: The main class that performs the clustering.
    - **`__init__(config)`**: Initializes the clustering engine with the application configuration.
    - **`cluster_by_content_similarity(bookmarks)`**: Clusters bookmarks based on the similarity of their text content using a TF-IDF vectorizer and K-means clustering.
    - **`cluster_by_domain_patterns(bookmarks)`**: Clusters bookmarks based on their domain names.
    - **`cluster_by_tags(bookmarks)`**: Clusters bookmarks based on their tag similarity using DBSCAN.
    - **`_find_optimal_cluster_count(data_matrix)`**: Finds the optimal number of clusters using the elbow method and silhouette analysis.
    - **`_create_content_clusters(...)`**: Creates `BookmarkCluster` objects from the results of the content-based clustering.
    - **`_create_tag_clusters(...)`**: Creates `BookmarkCluster` objects from the results of the tag-based clustering.
    - **`_extract_cluster_keywords(cluster_bookmarks)`**: Extracts representative keywords for a cluster.
    - **`_suggest_collection_name(cluster_bookmarks, keywords)`**: Suggests a collection name for a cluster based on its content and keywords.

### Content Analyzer (`content_analyzer.py`)

The `ContentAnalyzer` performs in-depth analysis of the content of each bookmark. It extracts topics, sentiment, keywords, and other useful metadata.

**Classes:**

- **`Topic`**: A data class that represents a discovered topic, including its name, keywords, and weight.
- **`SentimentScore`**: A data class that represents the sentiment analysis result.
- **`ContentAnalysisResult`**: A data class that encapsulates the results of a content analysis operation.
- **`ContentAnalyzer`**: The main class that performs the content analysis.
    - **`__init__(config)`**: Initializes the content analyzer with the application configuration.
    - **`train_models(documents)`**: Trains the TF-IDF vectorizer and LDA topic model on a corpus of documents.
    - **`analyze_content(title, content, url)`**: Performs a comprehensive analysis of the given content, including topic modeling, sentiment analysis, and keyword extraction.
    - **`_analyze_sentiment(text)`**: Analyzes the sentiment of a block of text.
    - **`_extract_topics(text)`**: Extracts topics from a block of text using the trained LDA model.
    - **`_classify_content_type(title, content, url)`**: Classifies the content into different types (e.g., "tutorial", "news", "documentation").
    - **`get_similar_content(target_text, candidate_texts, threshold)`**: Finds similar content from a list of candidates using cosine similarity.

### Network Analyzer (`network_analyzer.py`)

The `NetworkAnalyzer` builds a network graph of bookmarks to understand the relationships between them. This is useful for discovering communities and identifying influential bookmarks.

**Classes:**

- **`BookmarkNode`**: A data class that represents a bookmark node in the network.
- **`Community`**: A data class that represents a detected community of bookmarks.
- **`HubBookmark`**: A data class that represents an important hub bookmark.
- **`CollectionStructure`**: A data class that represents an optimized collection structure.
- **`NetworkAnalysisResult`**: A data class that encapsulates the results of a network analysis operation.
- **`NetworkAnalyzer`**: The main class that performs the network analysis.
    - **`__init__(similarity_threshold, min_community_size)`**: Initializes the network analyzer with the given parameters.
    - **`build_bookmark_network(bookmarks)`**: Builds a network graph from a list of bookmarks.
    - **`find_bookmark_communities()`**: Detects communities in the bookmark network using the Louvain algorithm.
    - **`identify_hub_bookmarks()`**: Identifies important hub bookmarks based on centrality measures.
    - **`suggest_collection_structure(communities)`**: Suggests an optimized collection structure based on the detected communities.
    - **`analyze_network(bookmarks)`**: Performs a complete network analysis on a list of bookmarks.

### Ollama Client (`ollama_client.py`)

The `OllamaClient` provides an interface to interact with the Ollama service for running large language models (LLMs). It can be used for tasks like generating summaries, suggesting categories, and more.

**Classes:**

- **`OllamaModel`**: A data class that represents an Ollama model.
- **`OllamaResponse`**: A data class that represents a response from the Ollama API.
- **`OllamaClient`**: The main class for interacting with the Ollama service.
    - **`__init__(host, model_name, auto_start, auto_pull)`**: Initializes the Ollama client with the given parameters.
    - **`ensure_server_running()`**: Ensures that the Ollama server is running, and starts it if it's not.
    - **`ensure_model_available(model_name)`**: Ensures that the specified model is available, and pulls it if it's not.
    - **`generate_response(prompt, model, system_prompt, temperature, max_tokens)`**: Generates a response from the Ollama model.
    - **`generate_bookmark_summary(title, content, url, max_length)`**: Generates a concise summary for a bookmark.
    - **`suggest_categories(title, content, url, existing_categories)`**: Suggests categories for a bookmark.
    - **`generate_smart_tags(title, content, url, existing_tags)`**: Generates intelligent tags for a bookmark.

### Similarity Engine (`similarity_engine.py`)

The `SimilarityEngine` is used to find similar bookmarks and detect duplicates. It uses sentence transformers to compute semantic embeddings of bookmark content.

**Classes:**

- **`SimilarityResult`**: A data class that represents the result of a similarity comparison.
- **`DuplicateGroup`**: A data class that represents a group of potentially duplicate bookmarks.
- **`SimilarityEngine`**: The main class that performs similarity analysis.
    - **`__init__(config)`**: Initializes the similarity engine with the application configuration.
    - **`compute_embeddings(texts)`**: Computes embeddings for a list of texts using a sentence transformer model.
    - **`compute_bookmark_embeddings(bookmarks)`**: Computes embeddings for a list of bookmarks.
    - **`find_similar_bookmarks(target_bookmark_id, bookmarks, limit)`**: Finds bookmarks that are similar to a given bookmark.
    - **`detect_duplicates(bookmarks)`**: Detects duplicate and near-duplicate bookmarks.
    - **`recommend_similar_bookmarks(user_bookmarks, candidate_bookmarks, limit)`**: Recommends bookmarks based on a user's existing bookmarks.

### Specialized Analyzers (`specialized_analyzers.py`)

The `specialized_analyzers` module provides a collection of analyzers that are specialized for different domains and content types, such as gaming, development, and research.

**Classes:**

- **`SpecializedAnalysisResult`**: A data class that represents the result of a specialized analysis.
- **`SpecializedAnalyzer`**: An abstract base class for all specialized analyzers.
- **`GamingAnalyzer`**: A specialized analyzer for gaming content.
- **`DevelopmentAnalyzer`**: A specialized analyzer for development and self-hosting content.
- **`ResearchAnalyzer`**: A specialized analyzer for research and educational content.
- **`SpecializedAnalysisEngine`**: The main engine that coordinates all specialized analyzers.
    - **`analyze_content(url, title, content)`**: Analyzes the given content with all applicable specialized analyzers.
    - **`get_best_analysis(url, title, content)`**: Returns the best analysis result based on confidence scores.

### Tag Predictor (`tag_predictor.py`)

The `TagPredictor` uses machine learning to predict relevant tags for bookmarks. It can be trained on existing bookmarks to learn the relationships between content and tags.

**Classes:**

- **`TagPrediction`**: A data class that represents a single tag prediction with a confidence score.
- **`TrainingResult`**: A data class that represents the result of a model training operation.
- **`TagPredictor`**: The main class that performs tag prediction.
    - **`__init__(config)`**: Initializes the tag predictor with the application configuration.
    - **`train_from_bookmarks(bookmarks)`**: Trains the tag prediction model from a list of bookmarks.
    - **`predict_tags(title, content, url, existing_tags)`**: Predicts tags for the given content.
    - **`_predict_with_model(text)`**: Predicts tags using the trained machine learning model.
    - **`_predict_with_patterns(text, url)`**: Predicts tags using pattern matching.
    - **`_predict_with_frequency(text)`**: Predicts tags based on their frequency in the training data.
    - **`update_with_feedback(text, predicted_tags, actual_tags)`**: Updates the model with user feedback.
