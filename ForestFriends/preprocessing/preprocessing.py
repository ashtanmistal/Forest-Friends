import os
import sys
import numpy as np
import pandas as pd
import pylas
import requests
from scipy.interpolate import griddata
from tqdm import tqdm
import utils
import os
from openai import OpenAI
import rasterio

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
data_dir = r"C:\Users\Ashtan Mistal\OneDrive - UBC\School\2023S\minecraftUBC\resources"

coniferous_families = [
    "pinaceae",  # Pine family
    "cupressaceae",  # Cypress family
    "araucariaceae",  # Araucaria family
    "podocarpaceae",  # Yellow-wood family
    "sciadopityaceae",  # Umbrella-pine family
    "cephalotaxaceae",  # Plum-yew family
    "taxaceae"  # Yew family
]


def get_tree_category_gpt(taxa):
    prompt = f"Please tell me if {taxa} is deciduous or coniferous. Respond with one word."

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo",
    )

    return response.choices[0].message.content.strip()


def get_tree_category_gbif(taxa):
    gbif_species_url = 'http://api.gbif.org/v1/species/match'
    params = {'name': taxa, 'kingdom': 'plants'}
    try:
        response = requests.get(gbif_species_url, params=params)
        if response.status_code == 200:
            data = response.json()
            family = data.get('family', '').lower()
            if family and family in coniferous_families:
                return 'coniferous'
            elif family:
                return 'deciduous'  # Too many different families to list and test for, so this is more of a catch-all
            else:
                return get_tree_category_gpt(taxa)
        else:
            raise ValueError(f"API request failed with status code {response.status_code}")
    except requests.RequestException as e:
        raise ValueError(f"An error occurred: {e}")


def get_tree_taxa_from_gpt(common_name):
    # This is a backup to the GBIF API, in case it fails.
    # it calls gpt-3 to get the taxon from the common name
    prompt = (f"Please provide the scientific name of the {common_name} tree. DO NOT RESPOND WITH ANYTHING ELSE "
              f"EXCEPT FOR THE SCIENTIFIC NAME.")

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo",
    )

    return response.choices[0].message.content.strip()


def get_tree_taxa(common_name):
    # Base URL for the GBIF species API
    gbif_species_url = 'http://api.gbif.org/v1/species/match'

    # Query parameters for the API request
    params = {'vernacularName': common_name, 'kingdom': 'plants'}

    try:
        # Sending a request to the GBIF API
        response = requests.get(gbif_species_url, params=params)
        if response.status_code == 200:
            data = response.json()

            # Extracting the scientific name
            scientific_name = data.get('scientificName')
            if scientific_name:
                return scientific_name
            else:
                return get_tree_taxa_from_gpt(common_name)
        else:
            raise ValueError(f"API request failed with status code {response.status_code}")
    except requests.RequestException as e:
        raise ValueError(f"An error occurred: {e}")


def populate_dataframe():
    # check if the processed csv file already exists
    # if it does, load it into a dataframe
    # if it doesn't, create it from the original csv file
    # if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
    #                                "ubcv_campus_trees_processed.csv")):
    #     df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
    #                                   "ubcv_campus_trees_processed.csv"))
    if os.path.exists(os.path.join(data_dir, "ubcv_campus_trees_processed.csv")):
        df = pd.read_csv(os.path.join(data_dir, "ubcv_campus_trees_processed.csv"))
    else:
        # read the original csv file
        # csv_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        #                              "ubcv_campus_trees.csv")
        csv_file_path = os.path.join(data_dir, "ubcv_campus_trees.csv")

        df = pd.read_csv(csv_file_path)

        # we care about the following columns:
        columns = [
            "TAXA",
            "COMMON_NAME",
            "TREE_TYPE",
            "LAT",
            "LONG"
        ]

        df = df[columns]
        # make new x and z columns
        df["X"] = None
        df["Z"] = None

        # Remove rows where both COMMON_NAME and TAXA are missing
        df.dropna(subset=['COMMON_NAME', 'TAXA'], how='all', inplace=True)
        # convert all tree types, taxons, and common names to lowercase
        df["TREE_TYPE"] = df["TREE_TYPE"].str.lower()
        df["TAXA"] = df["TAXA"].str.lower()
        df["COMMON_NAME"] = df["COMMON_NAME"].str.lower()
        # sort by taxon
        df.sort_values(by=["TAXA"])

        # save the processed dataframe to a csv file
        df.to_csv(os.path.join(data_dir, "ubcv_campus_trees_processed.csv"), index=False)

    for index, row in tqdm(df.iterrows(), total=len(df.index)):
        # If tree type is already known, skip
        # If TAXA is missing, fill it in with the GBIF API through the common name
        # Once TAXA is known (or if it was already known), determine the tree type
        # by making a call to the GBIF API
        # Fill in the tree type for all rows with the same taxon to avoid making
        # multiple calls to the API for the same taxon
        if pd.isna(row["TREE_TYPE"]):
            if pd.isna(row["TAXA"]):
                if not pd.isna(row["COMMON_NAME"]):
                    # Make a call to the GBIF API to get the taxon
                    # Fill in the taxon for all rows with the same common name
                    # to avoid making multiple calls to the API for the same common name
                    taxa = get_tree_taxa(row["COMMON_NAME"])
                    df.loc[df["COMMON_NAME"] == row["COMMON_NAME"], "TAXA"] = taxa
                else:
                    # If both the common name and taxon are missing, skip
                    raise ValueError(
                        "Both the common name and taxon are missing for this row - this should have been filtered out earlier")
            # Make a call to the GBIF API to get the tree type
            # Fill in the tree type for all rows with the same taxon
            # to avoid making multiple calls to the API for the same taxon
            tree_type = get_tree_category_gbif(row["TAXA"])
            df.loc[df["TAXA"] == row["TAXA"], "TREE_TYPE"] = tree_type
            # save the dataframe to a csv file to avoid losing progress
            df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                                   "ubcv_campus_trees_processed.csv"), index=False)
    # convert the latitude and longitude to x and z coordinates
    # and fill in the x and z columns
    x, z = utils.convert_lat_long_to_x_z(df["LAT"], df["LONG"], return_int=False)
    df["X"] = x
    df["Z"] = z

    # Remove rows where the tree type is still missing or unknown
    df.dropna(subset=['TREE_TYPE'], inplace=True)
    df = df[df["TREE_TYPE"] != "unknown"]

    # GPT-3 liked to put periods at the end of its answer :/
    df["TREE_TYPE"] = df["TREE_TYPE"].str.replace(".", "")
    df["TREE_TYPE"] = df["TREE_TYPE"].str.lower()
    df["TREE_TYPE"] = df["TREE_TYPE"].str.replace("evergreen", "coniferous")

    # assert 1) all tree types are either deciduous or coniferous
    #        2) there are no missing labels
    assert len(df["TREE_TYPE"].unique()) == 2
    assert not df["TREE_TYPE"].isna().any()
    # If there are these are likely stochastic errors caused by the various API calls.
    # Some manual adjustments may be required.

    # save the dataframe to a csv file
    df.to_csv(os.path.join(data_dir, "ubcv_campus_trees_processed.csv"), index=False)


if __name__ == "__main__":
    populate_dataframe()
    print("Preprocessing complete.")
