from collections import defaultdict
from typing import Iterator

from pheval.post_processing.post_processing import PhEvalDiseaseResult
from pydantic import BaseModel
from typing import List, Dict
import numpy as np


BEST_MATCH_SCORES = Dict[str, Dict[str, Dict[str, float]]]


class DiseaseScore(BaseModel):
    disease_id: str
    disease_name: str
    avg_best_match_score: float

class BestMatchScore(BaseModel):
    disease_id: str
    disease_name: str
    best_scores: list[float]

class TermSetPairWiseComparisonQuery:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.similarity_strategy = None
        self.hp_embeddings = self.data_processor.hp_embeddings
        self.disease_to_hps_from_omim = self.data_processor.disease_to_hps
        self._cosine_cache = defaultdict()

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def cached_cosine_similarity(self, input_hp: str, disease_hp: str) -> float:
        key = (input_hp, disease_hp)
        if key in self._cosine_cache:
            return self._cosine_cache[key]

        if input_hp not in self.hp_embeddings:
            raise ValueError(f"Embedding for phenotype {input_hp} not found in hp_embeddings.")
        if disease_hp not in self.hp_embeddings:
            raise ValueError(f"Embedding for disease phenotype {disease_hp} not found in hp_embeddings.")

        inp_embedding = self.hp_embeddings[input_hp]['embeddings']
        disease_embedding = self.hp_embeddings[disease_hp]['embeddings']
        similarity = self.cosine_similarity(inp_embedding, disease_embedding)
        self._cosine_cache[key] = similarity
        return similarity


    def termset_pairwise_comparison_disease_embeddings(self, hps: List[str], nr_of_results: int) -> Iterator[PhEvalDiseaseResult]:
        best_match_scores: BEST_MATCH_SCORES = defaultdict(lambda: {"best_scores": {}})

        for input_hp in hps:
            if input_hp not in self.hp_embeddings:
                print(f"Embedding for phenotype {input_hp} not found in hp_embeddings list: {hps}.")
                with open("missed_hps.txt", "a") as f:
                    f.write(f"{input_hp}\n")
                continue

            for disease_id, disease_data in self.disease_to_hps_from_omim.items():
                disease_name = disease_data["disease_name"]
                disease_phenotypes = disease_data["phenotypes"]

                # cosine similarities between input_hp and each disease-phenotype
                cosine_similarities = [
                    self.cached_cosine_similarity(input_hp, disease_hp)
                    for disease_hp in disease_phenotypes
                    if disease_hp in self.hp_embeddings
                ]

                # store
                if cosine_similarities:
                    best_match = max(cosine_similarities)
                    best_match_scores[disease_id]["disease_name"] = disease_name
                    best_match_scores[disease_id]["best_scores"][input_hp] = best_match

                    """
                    Example: best_match_scores structure

                    {
                        "OMIM:101600": {
                            "disease_name": "Marfan Syndrome",
                            "best_scores": {
                                "HP:0001250": 0.98,  # Best similarity score per input phenotype
                                "HP:0004322": 0.60
                            }
                        },
                        "OMIM:203400": {
                            "disease_name": "Ehlers-Danlos Syndrome",
                            "best_scores": {
                                "HP:0001250": 0.88,
                                "HP:0004322": 0.57
                            }
                        }
                    }
                    """

        for disease_id, scores in sorted(
                best_match_scores.items(),
                key=lambda item: np.mean(list(item[1]["best_scores"].values())),
                reverse=True
        )[:nr_of_results]:  # sorting directly in iteration

            avg_best_match_score = float(np.mean(list(scores["best_scores"].values())))

            yield PhEvalDiseaseResult(
                disease_identifier=disease_id,
                disease_name=scores.get("disease_name", "Unknown Disease"),
                score=avg_best_match_score
            )
