# Histograms for LEXTREME datasets

This directory contains images with histograms that depict the distribution of document length after applying tokenization with the five pretrained Tranformer-based models from the paper(```distilbert-base-multilingual-cased```,```microsoft/Multilingual-MiniLM-L12-H384```,```microsoft/mdeberta-v3-base```,```xlm-roberta-base```,```xlm-roberta-large```). The last bins, which also have slighter darker colors, show outliers, i.e. the number of documents that fall outside the 99<sup>th</sup> percentile. 

For each dataset there is a folder with histograms. For multilingual datasets we created a general histogram on the basis of the entire dataset and a histogram for each language.

```Histograms_for_datasets_with_hierarchical_models.jpg``` shows an overview of histograms for those datasets that had particularly long input documents and thus required hierarchical models.