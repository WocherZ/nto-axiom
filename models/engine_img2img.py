
import torch
import transformers
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

import io
import ast
import base64
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
import matplotlib.pyplot as plt

def get_model_info(model_ID: str) -> tuple:
    """
    A function that retrieves information about a model.
    
    Parameters:
        model_ID (str): The ID of the model to retrieve information for.
    
    Returns:
        tuple: A tuple containing the model, processor, and tokenizer.
    """
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_ID).to(device)
    processor = CLIPProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    return model, processor, tokenizer

class ImgToImg:
    def __init__(self):
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ID = "openai/clip-vit-base-patch32"
        self.model, self.processor, self.tokenizer = self.get_model_info_()
    
    def get_model_info_(self) -> tuple:
        """
        A function that retrieves model information, including the model itself, processor, and tokenizer.
        
        Returns:
            tuple: A tuple containing the model, processor, and tokenizer.
        """
        model = CLIPModel.from_pretrained(self.model_ID).to(self.device)
        processor = CLIPProcessor.from_pretrained(self.model_ID)
        tokenizer = CLIPTokenizer.from_pretrained(self.model_ID)
        return model, processor, tokenizer
    
    def get_single_text_embedding_(self, text: str) -> np.ndarray:
        """
        Generate a single text embedding.

        Parameters:
            text (str): The input text to be embedded.

        Returns:
            numpy.ndarray: The embedding of the input text as a numpy array.
        """ 
        inputs = self.tokenizer(text, return_tensors = "pt").to(self.device)
        text_embeddings = self.model.get_text_features(**inputs)
        embedding_as_np = text_embeddings.cpu().detach().numpy()
        return embedding_as_np
    
    def get_single_image_embedding_(self, my_image: Image) -> np.ndarray:
        """
        Get the embedding of a single image.

        Parameters:
            my_image: The image for which to retrieve the embedding.

        Returns:
            numpy.ndarray: The embedding_as_np: The embedding of the input image as a numpy array.
        """
        image = self.processor(text = None,
                               images = my_image, 
                               return_tensors="pt")["pixel_values"].to(self.device)
        
        embedding = self.model.get_image_features(image)
        embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np
    
    def get_top_N_images_(self, 
                         query: Image, 
                         data: pd.DataFrame, 
                         spots: pd.DataFrame, 
                         top_K: int = 100,
                         search_criterion: str = "text") -> pd.DataFrame:
        """
        Retrieve top N similar images based on the given query.
        
        Parameters:
            query (Union[str, List[float]]): The query for image similarity search. It can be either text or image embedding.
            data (pd.DataFrame): DataFrame containing image data with columns including 'img_embeddings'.
            spots (pd.DataFrame): DataFrame containing information about spots with a column named 'Name'.
            top_K (int, optional): Number of top similar images to retrieve. Defaults to 100.
            search_criterion (str, optional): The criterion for search, either 'text' or 'image'. Defaults to 'text'.
            
        Returns:
            pd.DataFrame: DataFrame containing information about top similar images.
        """  
        # Text to image Search
        if(search_criterion.lower() == "text"):
            query_vect = self.get_single_text_embedding_(query)

        # Image to image Search
        else:
            query_vect = self.get_single_image_embedding_(query)

        # Run similarity Search
        column_names = [f'embedding_{i}' for i in range(512)]
        
        data["cos_sim"] = data[column_names].values.apply(lambda x: cosine_similarity(query_vect, x))
        data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])

        filters = []
        n = 10
        most_similar = pd.DataFrame(data.sort_values(by='cos_sim', ascending=False))
        places = pd.DataFrame(columns= ["name", "image", "img_embeddings","pil", "cos_sim" ])
        # sights = pd.DataFrame(columns= ["WikiData","Name","Kind", "City","Rate","Lon", "Lat", "russian_captions" ])
        
        for index, row in most_similar.iterrows():
            if row["name"].lower() not in filters:
                places.loc[index] = row
                # sights.loc[index] = spots.where(spots["Name"] == row["name"]).iloc[0]
                filters.append(row["name"].lower())
            if len(places) == n:
                break
        return filters
    
    def plot_images_by_side_(self,
                            top_images: pd.DataFrame, 
                            n_row: int, 
                            n_col: int):
        """
        Generate a plot displaying images side by side.

        Parameters:
            top_images (DataFrame): A DataFrame containing images to be plotted.
            n_row (int): Number of rows for the plot grid.
            n_col (int): Number of columns for the plot grid.
        """
        index_values = list(top_images.index.values)
        list_images = [top_images.iloc[idx].pil for idx in index_values] 
        similarity_score = [top_images.iloc[idx].cos_sim for idx in index_values] 

        _, axs = plt.subplots(n_row, n_col, figsize=(15, 15))
        axs = axs.flatten()
        for img, ax,  sim_score in zip(list_images, axs,  similarity_score):
                ax.imshow(img)
                sim_score = 100*float("{:.2f}".format(sim_score))
                ax.title.set_text(f"Similarity: {sim_score}%")
        plt.show()

    def get_total_embds_(self, 
                        df: pd.DataFrame, 
                        df_places: pd.DataFrame, 
                        img_column: str, 
                        captions_column: str):
        """
        Get total embeddings for images based on image embeddings and captions embeddings.

        Parameters:
            df (pd.DataFrame): DataFrame containing image data.
            df_places (pd.DataFrame): DataFrame containing information about places with a column named 'Name'.
            img_column (str): Name of the column containing image data.
            captions_column (str): Name of the column containing captions data.

        """
        temp_lst = []

        for index, row in df.iterrows():
            temp_lst.append(self.get_single_image_embedding_(row["pil"])[0])

        temp_lst = np.array(temp_lst)
        
        for i in range(temp_lst.shape[1]):
            df[f"embedding_{i}"] = temp_lst[:, i]

        # df["img_embeddings"] = np.array(temp_lst)
    
        # temp_lst = []
        # for index, row  in df_places.iterrows():
        #     temp_vec = 0
        #     for j in row[captions_column].split(" "):
        #             temp_vec += get_single_text_embedding(j)
        #     temp_lst.append(temp_vec)
        # df_places["captions_embds"] = temp_lst
        # temp_lst = []
        # for index, row in df.iterrows():
        #     temp_lst.append(df_places[df_places["Name"] == row["name"]]["captions_embds"])
            
        # df["captions_embds"] = temp_lst
        # df["total"] = df["captions_embds"] + df["img_embeddings"]
    
    def dataToJpg_(self, data, source):
        """
        Convert data to jpg images and store them in the given source.
        
        Parameters:
            data (list): The input data containing images and their metadata.
            source (str): The destination directory to store the jpg images.
        """
        for i in data:
            temp = []
            for j in range(len(i["image"])):
                temp.append(Image.open(io.BytesIO(base64.b64decode(list(i["image"])[j]))))
            i["pil"] = temp
     

def get_single_text_embedding(text, model, tokenizer):
    """
    Generate a single text embedding using a given model and tokenizer.

    Parameters:
        text (str): The input text to generate the embedding for.
        model: The text embedding model to use.
        tokenizer: The tokenizer to preprocess the input text.

    Returns:
        numpy.ndarray: The text embedding as a NumPy array.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    inputs = tokenizer(text, return_tensors = "pt").to(device)
    text_embeddings = model.get_text_features(**inputs)
    embedding_as_np = text_embeddings.cpu().detach().numpy()

    return embedding_as_np

def get_single_image_embedding(my_image, model, processor):
    """
    Generate the embedding for a single image using the provided model and processor.

    Parameters:
        my_image (PIL.Image): The image to generate the embedding for.
        model: The image embedding model to use.
        processor: The processor to preprocess the input image.

    Returns:
        numpy.ndarray: The image embedding as a NumPy array.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = processor(text = None,
                      images = my_image, 
                      return_tensors="pt")["pixel_values"].to(device)

    embedding = model.get_image_features(image)
    embedding_as_np = embedding.cpu().detach().numpy()

    return embedding_as_np

def get_top_N_images(query, data, top_K=100, search_criterion="text"):
    """
    A function to get the top N images based on a query and data using text or image search criteria.
    
    Parameters:
        query (str): The query to search for.
        data (pd.DataFrame): The data to search in.
        top_K (int, optional): The number of top images to return. Defaults to 100.
        search_criterion (str, optional): The search criterion to use. Can be "text" or "image". Defaults to "text".

    Returns:
        pd.DataFrame: The top N images based on the query.
    """
    # Text to image Search
    if(search_criterion.lower() == "text"):
      query_vect = get_single_text_embedding(query)

    # Image to image Search
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
    places = pd.DataFrame(columns= ["name", "image", "img_embeddings", "cos_sim" ])
    
    for index, row in most_similar.iterrows():
        if row["name"].lower() not in filters:
            places.loc[index] = row
            filters.append(row["name"].lower())
        if len(places) == n:
            break
    
    return places[revevant_cols].reset_index()

def plot_images_by_side(top_images: pd.DataFrame):
  """
  Generate a side-by-side plot of images with their similarity scores.

  Parameters:
      top_images (pd.DataFrame): A DataFrame containing the top images and their similarity scores.
  """
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


def get_total_embds(df: pd.DataFrame, 
                     df_places: pd.DataFrame, 
                     img_column: str,
                     captions_column: str):
    """
    This function calculates the total embeddings for images and captions in the given dataframes.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing image data.
        df_places (pd.DataFrame): DataFrame containing information about places with a column named 'Name'.
        img_column (str): Name of the column containing image data.
        captions_column (str): Name of the column containing captions data.

    Returns:
        pd.DataFrame: DataFrame containing total embeddings for images and captions.
    """
    temp_lst = []

    for index, row in df.iterrows():
        temp_lst.append(get_single_image_embedding(row["pil"])[0])

    temp_lst = np.array(temp_lst)
    
    for i in range(temp_lst.shape[1]):
        df[f"embedding_{i}"] = temp_lst[:, i]
    return df

def getData(names: list, places: pd.DataFrame):
    """
    A function that filters entries from a DataFrame based on names and returns a new DataFrame.
    
    Parameters:
        names (list): A list of names to filter the DataFrame.
        places (DataFrame): The DataFrame containing entries to filter.
    
    Returns:
        DataFrame: A new DataFrame containing the filtered entries.
    """
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