"""
Cap3D Filtering Module

This module performs the following tasks:
- Reads a dataset containing caption descriptions from a CSV file.
- Tokenizes and processes the text data to create a set of keywords for filtering.
- Filters the dataset based on specified keywords and categories.
- Saves the filtered and excluded results as JSON files for further use.

Constants and Configuration:
- The module retrieves configuration from a YAML file, typically named "filter_dataset.yaml".
- Constants include file encoding, categories to exclude, and various paths for data input and output.

Data Processing Workflow:
1. Import data from a CSV file containing caption descriptions.
2. Tokenize and process the descriptions to create a vocabulary.
3. Load a set of keywords from a JSON file for filtering.
4. Filter the dataset based on the presence of keywords or their plurals in the descriptions.
5. Save the filtered and excluded datasets to specified paths in JSON format.

Dependencies:
- Pandas for data manipulation and analysis.
- NLTK for text tokenization.
- PyYAML for reading constants from a YAML configuration file.

Usage:
- Ensure the required files and directories are set up as specified in the YAML configuration file.
- Run the script with a command-line argument specifying the path to the YAML configuration file.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Union

import nltk
import pandas as pd
import yaml
from nltk.tokenize import word_tokenize

nltk.download("punkt")


def main(config_path: Union[str, Path]) -> None:
    """
    Main function to filter a dataset based on a configuration YAML file.

    Reads constants and configuration from the given YAML file, performs the following tasks:
    - Reads a CSV file with caption descriptions.
    - Tokenizes and processes the descriptions to create a vocabulary.
    - Loads a set of keywords for filtering from a JSON file.
    - Filters the dataset based on the presence of specified keywords or their plurals.
    - Saves the filtered and excluded datasets to specified paths in JSON format.

    Parameters:
    - config_path (str or Path): Path to the YAML configuration file.
    """
    print("Initiating Cap3D filtering process")

    ### Constants
    # Read constants from the YAML file
    with open(config_path, "r", encoding="utf-8") as file:
        constants = yaml.safe_load(file)

    ENCODING: str = constants["encoding"]
    EXCLUDED_CATEGORIES: List[str] = constants["excluded_categories"]
    PATHS = constants["paths"]

    PATH_TO_CAPTIONS: Union[str, Path] = PATHS["captions"]
    PATH_TO_KEYWORDS_BY_CATEGORY: Union[str, Path] = PATHS["keywords_by_category"]

    PATH_TO_INCLUDED_IDS: Union[str, Path] = PATHS["included_ids"]
    PATH_TO_EXCLUDED_IDS: Union[str, Path] = PATHS["excluded_ids"]

    print(f"ENCODING: {ENCODING}")
    print(f"EXCLUDED_CATEGORIES: {EXCLUDED_CATEGORIES}")
    print(f"PATH_TO_CAPTIONS: {PATH_TO_CAPTIONS}")
    print(f"PATH_TO_KEYWORDS_BY_CATEGORY: {PATH_TO_KEYWORDS_BY_CATEGORY}")
    print(f"PATH_TO_INCLUDED_IDS: {PATH_TO_INCLUDED_IDS}")
    print(f"PATH_TO_EXCLUDED_IDS: {PATH_TO_EXCLUDED_IDS}")
    print()

    ### Data import
    print(f"Reading captions at '{PATH_TO_CAPTIONS}'")
    df = pd.read_csv(
        PATH_TO_CAPTIONS,
        header=None,
        names=["id", "desc"],
    )
    print("Successfully read captions")
    print()

    ### Tokenisation
    # Convert descriptions to lowercase
    print("Converting captions to lowercase")
    df["desc tokens"] = df["desc"].apply(
        lambda desc: desc.lower().replace("/", " / ").replace("-", " - ")
    )
    print("Successfully converted captions to lowercase")
    print()

    # Tokenize each description
    print("Tokenising captions")
    df["desc tokens"] = df["desc tokens"].apply(word_tokenize)
    print("Successfully tokenised captions")
    print()

    ### Vocabulary extraction
    # Create the vocabulary
    print("Creating vocabulary")
    vocab = set()

    for tokens in df["desc tokens"]:
        # Update the vocabulary
        vocab.update([token.replace("'", "") for token in tokens])

    print("Successfully created vocabulary")
    print()

    ### Load keywords by category
    print(f"Loading keywords by category at '{PATH_TO_KEYWORDS_BY_CATEGORY}'")
    with open(PATH_TO_KEYWORDS_BY_CATEGORY, "rb") as f:
        keywords_dict = json.load(f)

    print("Successfully loaded keywords by category")
    print()

    print(f"Creating keywords")
    keywords = set(
        [
            item.lower()
            for category, sublist in keywords_dict.items()
            for item in sublist
            if category not in EXCLUDED_CATEGORIES
        ]
    )
    print("Successfully created keywords")
    print()

    ### Filter dataset
    def check_keywords(tokens: List[str]) -> bool:
        """
        Check if any token matches a specified keyword or its plural.

        Parameters:
        - tokens (List[str]): List of tokens to check.

        Returns:
        - bool: True if any token matches a keyword or its plural, False otherwise.
        """
        return any(
            ((token in keywords) or (f"{token}s" in keywords)) for token in tokens
        )

    # Determine if rows should be included
    df["include"] = df["desc tokens"].apply(lambda tokens: check_keywords(tokens))

    print("Filtering dataset")
    filtered_df = df[df["include"] == True].drop(columns=["include"])
    excluded_df = df[df["include"] == False].drop(columns=["include"])
    print("Successfully filtered dataset")
    print()

    ### Save filtered dataset
    print(
        f"Making directories ot save processed dataset ids: '{os.path.dirname(PATH_TO_INCLUDED_IDS)}', '{os.path.dirname(PATH_TO_EXCLUDED_IDS)}'"
    )
    os.makedirs(os.path.dirname(PATH_TO_INCLUDED_IDS), exist_ok=True)
    os.makedirs(os.path.dirname(PATH_TO_EXCLUDED_IDS), exist_ok=True)

    print("Saving filtered ids")
    with open(PATH_TO_INCLUDED_IDS, "w", encoding=ENCODING) as f:
        json.dump(filtered_df["id"].tolist(), f, indent=4)
    print("Saved filtered ids")

    print("Saved excluded ids")
    with open(PATH_TO_EXCLUDED_IDS, "w", encoding=ENCODING) as f:
        json.dump(excluded_df["id"].tolist(), f, indent=4)
    print("Saved excluded ids")
    print()

    print("Cap3D filtering process successfully completed")


# Entry point for the script
if __name__ == "__main__":
    # Argument parser for command-line input
    parser = argparse.ArgumentParser(
        description="Script to filter a dataset based on specified keywords and save the results."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="filter_dataset.yaml",
        help="Path to the YAML configuration file containing constants and paths.",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function with the config path
    main(args.config_path)
