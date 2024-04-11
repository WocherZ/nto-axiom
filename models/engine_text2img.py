import ruclip
import torch
from huggingface_hub import hf_hub_url, cached_download
from ruclip import model, processor, predictor
from ruclip.model import CLIP
from ruclip.processor import RuCLIPProcessor
from ruclip.predictor import Predictor
import pandas as pd
from PIL import Image
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class RuClipTopK:
    def __init__(self, device, topK, templates):
        self.topK = topK
        self.device = device
        self.clip = CLIP.from_pretrained("ruclip-vit-base-patch32-384").eval().to(device)
        self.clip_processor = RuCLIPProcessor.from_pretrained("ruclip-vit-base-patch32-384")
        self.predictor = ruclip.Predictor(self.clip, 
                                          self.clip_processor, 
                                          device, 
                                          bs=8, 
                                          templates=templates)
    
    def get_single_text_embedding_(self, text):
        """
        Get the text embedding for a single text input.

        Parameters:
            text (str): The text input for which the embedding needs to be generated.

        Returns:
            numpy array: The text embedding as a numpy array.
        """
        with torch.no_grad():
            text_embeddings = self.predictor.get_text_latents([text])
     
        embedding_as_np = text_embeddings.cpu().detach().numpy()

        return embedding_as_np
    
    def get_single_image_embedding_(self, image):
        """
        Function to get the embedding of a single image.

        Args:
            image: The input image for which the embedding needs to be obtained.

        Returns:
            numpy array: The embedding of the input image as a NumPy array.
        """
        with torch.no_grad():
            embedding = self.predictor.get_image_latents([image])
            embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np
    
    def get_top_N_images_(self, 
                         query, 
                         data, 
                         top_K=100):
        """
        Generate the top N images based on a query and a dataset.

        Parameters:
            query (str): The query string to search for.
            data (DataFrame): The dataset containing image embeddings.
            top_K (int): The number of top images to retrieve (default is 100).

        Returns:
            list: A list of top N image names.
        """
        query_vect = self.get_single_text_embedding_(query)
        revevant_cols = ["name","image", "cos_sim"]
        data["cos_sim"] = data["img_embeddings"].apply(lambda x: cosine_similarity(query_vect, x))
        data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])
        filters = []
        n = self.topK
        most_similar = pd.DataFrame(data.sort_values(by='cos_sim', ascending=False))
        places = pd.DataFrame(columns= ["name", "image", "img_embeddings", "cos_sim"])
        for index, row in most_similar.iterrows():
            if row["name"].lower() not in filters:
                places.loc[index] = row
                filters.append(row["name"].lower())
            if len(places) == n:
                break;       
        return filters
    
    def plot_images_by_side_(self, top_images, n_row, n_col):
        """
        Generate a plot displaying images side by side based on the input top_images dataframe.
        
        Parameters:
            top_images (DataFrame): The input DataFrame containing the top images to be plotted.
            n_row (int): Number of rows in the plot grid.
            n_col (int): Number of columns in the plot grid.
        """
        index_values = list(top_images.index.values)
        list_images = [top_images.iloc[idx].image for idx in index_values] 
        similarity_score = [top_images.iloc[idx].cos_sim for idx in index_values] 
        _, axs = plt.subplots(n_row, n_col, figsize=(15, 15))
        axs = axs.flatten()
        for img, ax,  sim_score in zip(list_images, axs,  similarity_score):
            ax.imshow(img)
            sim_score = 100*float("{:.2f}".format(sim_score))
            ax.title.set_text(f"Similarity: {sim_score}%")
        plt.show()

    def get_all_images_embedding_(self, df):
        """
        Get embeddings for all the images in the input dataframe and add them as a new column.
        
        Parameters:
            df (DataFrame): The input dataframe containing the images.

        Returns:
            DataFrame: The input dataframe with the new column added.
        """
        df["img_embeddings"] = df[str("image")].apply(self.get_single_image_embedding_())
        return df
    
    def get_total_embds_(self,df, 
                        df_places, 
                        img_column, 
                        captions_column):
        """
        Generate and assign image embeddings and captions embeddings to the given DataFrames.
        
        Parameters:
            df (DataFrame): The input dataframe containing the images.
            df_places (DataFrame): The input dataframe containing the captions.
            img_column (str): The name of the column containing the images.
            captions_column (str): The name of the column containing the captions.

        Returns:
            DataFrame: The input dataframe with the new columns added.
        """
        temp_lst = []
        for index, row in df.iterrows():
            temp_lst.append(self.get_single_image_embedding_(row["image"]))
        df["img_embeddings"] = temp_lst
        temp_lst = []
        for index, row  in df_places.iterrows():
            temp_vec = 0
            for j in row[captions_column].split(" "):
                    temp_vec += self.get_single_text_embedding_(j)
            temp_lst.append(temp_vec)
        df_places["captions_embds"] = temp_lst
        temp_lst = []
        for index, row in df.iterrows():
            temp_lst.append(df_places[df_places["Name"] == row["name"]]["captions_embds"])
      
        df["captions_embds"] = temp_lst
        return df
    
def get_single_text_embedding(text): 
    with torch.no_grad():
        text_embeddings = predictor.get_text_latents([text])
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np
    
def get_single_image_embedding(image):
    with torch.no_grad():
        embedding = predictor.get_image_latents([image])
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np

def get_top_N_images(query, data, top_K=100, search_criterion="text"):
    if(search_criterion.lower() == "text"):
        query_vect = get_single_text_embedding(query)
    else: 
        query_vect = get_single_image_embedding(query )
    # Relevant columns
    revevant_cols = ["image", "cos_sim"]
    # Run similarity Search
    data["cos_sim"] = data["img_embeddings"].apply(lambda x: cosine_similarity(query_vect, x))
    data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])
    filters = []
    n = 10
    most_similar = pd.DataFrame(data.sort_values(by='cos_sim', ascending=False))
    places = pd.DataFrame(columns= ["name", "image", "img_embeddings", "cos_sim"])
    for index, row in most_similar.iterrows():
        if row["name"].lower() not in filters:
            places.loc[index] = row;
            filters.append(row["name"].lower())
        if len(places) == n:
            break;
    print(filters)        
    return places[revevant_cols].reset_index()

def get_all_images_embedding(df):
    df["img_embeddings"] = df[str("image")].apply(get_single_image_embedding)
    return df

def plot_images_by_side(top_images):
    index_values = list(top_images.index.values)
    print(len(index_values))
    list_images = [top_images.iloc[idx].image for idx in index_values] 

    similarity_score = [top_images.iloc[idx].cos_sim for idx in index_values] 

    n_row = 5
    n_col = 2
    _, axs = plt.subplots(n_row, n_col, figsize=(15, 15))
    axs = axs.flatten()
    for img, ax,  sim_score in zip(list_images, axs,  similarity_score):
        ax.imshow(img)
        sim_score = 100*float("{:.2f}".format(sim_score))
        ax.title.set_text(f"Similarity: {sim_score}%")
    plt.show()

def get_total_embds(df, df_places, img_column, captions_column):
    temp_lst = []
    for index, row in df.iterrows():
        temp_lst.append(get_single_image_embedding(row["image"]))
    df["img_embeddings"] = temp_lst
    temp_lst = []
    for index, row  in df_places.iterrows():
        temp_vec = 0
        for j in row[captions_column].split(" "):
                temp_vec += get_single_text_embedding(j)
        temp_lst.append(temp_vec)
    df_places["captions_embds"] = temp_lst
    temp_lst = []
    for index, row in df.iterrows():
        temp_lst.append(df_places[df_places["Name"] == row["name"]]["captions_embds"])
    df["captions_embds"] = temp_lst
    return df

def getData(names, places):
    result_df = pd.DataFrame(columns= ["WikiData","Name","Kind", "City","Rate","Lon", "Lat", "russian_captions" ])
    for name in names:
        # Find entries with the given name in the 'Name' column of the DataFrame (case-insensitive)
        entries = places[places["Name"].str.lower() == name.lower()]
        
        # If entries are found, append them to the result DataFrame
        if not entries.empty:
            result_df = pd.concat([result_df, entries])
        else:
            print(name)
    
    return result_df