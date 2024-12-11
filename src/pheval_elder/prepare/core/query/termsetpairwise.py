from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

from pheval_elder.prepare.core.store.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
from pheval_elder.prepare.core.collections.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from pheval_elder.prepare.core.collections.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService


class TermSetPairWiseComparisonQuery:
    data_processor: DataProcessor
    db_manager: ChromaDBManager
    average_llm_embedding_service: DiseaseAvgEmbeddingService
    weighted_average_llm_embedding_service: DiseaseWeightedAvgEmbeddingService
    similarity_strategy = None,

    def __post_init__(self):
        self.db_manager = self.db_manager
        self.data_processor = self.data_processor
        self.similarity_strategy = self.similarity_strategy
        self.hp_embeddings = self.data_processor.hp_embeddings
        self.disease_to_hps_from_omim = self.data_processor.disease_to_hps_with_frequencies
        self.disease_service = self.average_llm_embedding_service
        self.disease_weighted_service = self.weighted_average_llm_embedding_service

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def termset_pairwise_comparison_on_weighted_disease_embeddings(
        self,
        hps: List[str],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:

        phenotype_to_disease_scores: Dict[str, Dict[str, float]]= defaultdict(dict)
        disease_avg_scores: Dict[str, List[float]] = defaultdict(list)
        final_disease_scores: Dict[str, float] = defaultdict()

        for phenotype in hps:
            phenotype_embedding = self.hp_embeddings[phenotype]
            disease_cosines = []
            for disease, disease_phenotype in self.disease_to_hps_from_omim.items():
                disease_phenotype_embedding = self.hp_embeddings[disease_phenotype]
                cosine_similarity = self.cosine_similarity(phenotype_embedding, disease_phenotype_embedding)
                disease_cosines.append(cosine_similarity)

                # avg cosim for this disease
                avg_cosine = float(np.mean(disease_cosines))
                phenotype_to_disease_scores[disease][phenotype] = avg_cosine
                disease_avg_scores[disease].append(avg_cosine)

                final_disease_scores: Dict[str, float] = {
                    disease: float(np.mean(scores)) for disease, scores in disease_avg_scores.items()
                }

        return dict(phenotype_to_disease_scores), final_disease_scores
