from typing import List, Tuple
import re
from tqdm import tqdm

import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from oaklib import get_adapter
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection

class HPDescription:
    def __init__(self, hp_id: str, label: str, description: str):
        self.hp_id = hp_id
        self.label = label
        self.description = description

class HPDescriptionGenerator:
    def __init__(self, ontology_adapter: str, llm: OpenAI, prompt_template: str):
        self.adapter = get_adapter(ontology_adapter)
        self.llm = llm
        self.template = PromptTemplate(
            input_variables=["terms_list"],
            template=prompt_template,
        )

    def fetch_hp_terms(self) -> List[Tuple[str, str]]:
        try:
            curies = self.adapter.all_entity_curies()
            return [(curie, self.adapter.get_label_by_curie(curie)) for curie in tqdm(curies, desc='Fetching Terms')]
        except Exception as e:
            print(f"Error fetching terms: {e}")
            return []

    def generate_bulk_descriptions(self, hp_terms: List[Tuple[str, str]]) -> List[HPDescription]:
        terms_text = "\n".join([f"HP ID: {hp_id}, Label: {label}" for hp_id, label in hp_terms])
        prompt = self.template.render(terms_list=terms_text)
        response = self.llm(prompt)
        return self.parse_descriptions(response, hp_terms)

    def parse_descriptions(self, response: str, hp_terms: List[Tuple[str, str]]) -> List[HPDescription]:
        descriptions = []
        description_lines = response.strip().split('\n')
        pattern = r"HP ID: (.*), Description: (.*)"
        for line, (hp_id, label) in zip(description_lines, hp_terms):
            match = re.match(pattern, line)
            if match:
                descriptions.append(HPDescription(hp_id, label, match.group(2)))
        return descriptions

    def generate_descriptions(self, batch_size=50) -> List[HPDescription]:
        hp_terms = self.fetch_hp_terms()
        all_descriptions = []
        for i in tqdm(range(0, len(hp_terms), batch_size), desc='Processing Batches'):
            batch_terms = hp_terms[i:i + batch_size]
            batch_descriptions = self.generate_bulk_descriptions(batch_terms)
            all_descriptions.extend(batch_descriptions)
        return all_descriptions

class HPDescriptionStorage:
    def __init__(self, chroma_collection: Collection):
        self.collection = chroma_collection

    def store_descriptions(self, descriptions: List[HPDescription]):
        for desc in tqdm(descriptions, desc='Storing Descriptions'):
            self.collection.add(
                documents=[desc.description],
                metadatas=[{"hp_id": desc.hp_id, "label": desc.label}],
                ids=[desc.hp_id],
            )

def main():
    llm = OpenAI(temperature=0.7)
    prompt_template = """
    Generate detailed descriptions for the following HP terms:
    {terms_list}

    Description:
    """
    generator = HPDescriptionGenerator("sqlite:obo:hp", llm, prompt_template)
    descriptions = generator.generate_descriptions()

    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="hp_descriptions_db"
    ))
    collection = chroma_client.get_or_create_collection(
        name="hp_descriptions",
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
    )
    storage = HPDescriptionStorage(collection)
    storage.store_descriptions(descriptions)

if __name__ == "__main__":
    main()

