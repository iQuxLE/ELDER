
# ELDER
Elder is an algorithm that uses text-embeddings for differential diagnosis tasks. It will take any list of `phenotype_term`(`hpo_id`) as input and queries a vector database 
consisting of Diseases and the average embeddings of its corresponding phenotype terms.

Those embeddings are generated by [curate-gpt](https://github.com/monarch-initiative/curate-gpt). To use it a user needs to download the `ont-hp` collection.
Furthermore you can use the `phenotype.hpoa` file to generate the DiseaseAverageCollection that contains an embedding for each disease or the `hpoa` collection created by curate-gpt.

By using [pheval](https://github.com/monarch-initiative/pheval) you can compare those results to [Exomiser](https://github.com/exomiser/Exomiser). Therefore, there is the pheval_elder direcoty.
To do the comparison to the full LIRICAL dataset you will need to take the data from the [pheval-repo](https://github.com/monarch-initiative/pheval) from `/corpora/phenopackets`. Than use `phevals` PhenopacketUtils to iterate over those.

## Methods
1. Averaging each Disease into an Embedding space, by averaging the corresponding HP terms-embeddings given by curateGPT using known HPOA data.
There is different models to take. 'text-embedding-3-small', 'text-embedding-3-large', and 'text-embedding-ada-002'. The current repo uses ada-002, currently I am implementing data from large. Large is the most promissing, by now and pushes the accuracy by 5% on average.
     
3. Clustering each Disease into their relevant organ systems and average all HP terms in one organ system, to get a relevant representation of each organ_system inside a disease        represented by the set of HP terms that cover this organ system. If HP terms do not cover all organ systems this vector will be multiplied by the embedding of the organ system * -1 for the whole length of the vector.

4. Weighting the embeddings by the frequency of each phenotype observed in the `phenotype.hpoa` file from the Jax Lab.

5. Creating new embeddings by implementing a new documents embedder into curate-gpt, on which I am working right now. This would incorporate more knowledge into the embedding space instead of just the HP definitions. Updated documents will incorporate the frequencies to any known disease, and enabling curing by embeddings but also text and a scoring system for both. It will take some time to built it without losing data as the token limit for some HPs is reached  very fast when more than 50 diseases are connected. However, a simple averaging batches approach should help for an experiment. Additionally it will be helpful to intergrate information about papers that describe the HPs or diseases and more relations in the DAG, so that the LLM knows what is connected.

The following table contains data of comparisons between the models and also in comparison to prompring. Top 1 to Top 10 shows how much diseases are detected correctly from a cohort of 385 patients with known rare diseases from a set of known phenotypes.

<img width="899" alt="Screenshot 2024-05-01 at 16 15 11" src="https://github.com/iQuxLE/ELDER/assets/70138474/bab48b36-2fd6-4d6b-bdb1-2c0e4bba1a71">


It is visible that 'text-embedding-3-large' brings the most value in comparison to SOTA tools like Exomiser. But also it is important to notice, that prompting with GPT-4 Turbo in comparison to 3.5 already doubles the accuracy. In comparison of prompting, using embeddings can even bring more value (up to 120% performance increase).However currently we do not have access to embedding models from GPT-4 Turbo. But we can add more information into the embeddable text and boost the performance, over time so that we end up with a better phenotype driven disease matching algorithm than using semantic similarity of phenotipyc data (Exomiser) and than finding the average information content of MICA (most informative common ancestor) inside the ontology. Just looking at the data with the current models the approach is very promising. Additionally you would not have to worry about hallucinations by self extracting the embeddable text, making the best use of current LLM models.


### Run Pheval-Runner
`poetry shell`
`poetry install`
`pheval run -i . -t . -r 'ElderPhevalRunner' -o .`


### Run Elder only

`python -m main.main`


### Requirements
By now needs the generated embeddings via [curate-gpt](https://github.com/iQuxLE/curate-gpt) as explained above.
