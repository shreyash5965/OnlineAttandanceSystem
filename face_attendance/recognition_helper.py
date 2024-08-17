import cv2
import numpy as np
import pandas as pd

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

#similarity and distance calculation
from sklearn.metrics import pairwise
import redis

# connect to redis database
hostname = 'redis-18992.c258.us-east-1-4.ec2.redns.redis-cloud.com'
port = '18992'
password = 'nUoPu0pKGDk95n5zatr1N1DgTDNmnAuG'
r = redis.StrictRedis(host=hostname,
               port= port, password=password)

# configer face analysis model
app_sc = FaceAnalysis(name='buffalo_sc',providers=['CPUExecutionProvider'])
app_sc.prepare(ctx_id=0,det_size=(640,640))


# name searching algorithm
def mL_search_algorithm(df,feature_column,test_data,threshold=0.5):
    """
    Perform a machine learning search.

    Parameters:
    -----------
    df : pandas DataFrame
    Original dataset of metadata of training images.

    feature_column : str
    Name of the column containing features usually 'Facial_Features'

    test_data : array-like
    metadata of test_iamge for similarity calculation.

    threshold : float, optional (default=0.5)
    Threshold for similarity above which a match is considered.

    Returns:
    --------
    tuple (str, str)
    Name and role of the person with highest similarity.
    ('Unknown', 'Unknown') if no match is found above threshold.
    """

    # 1. copy the dataframe and prepare for next step
    df_temp = df.copy()

    # 2. get face embeding from dataframe and prepare X(pretrained data) and y(test_data)
    X_list = df[feature_column].tolist()
    X = np.asarray(X_list)

    # reshape data to 1 row and as many columns as necessary
    y = test_data.reshape(1,-1)

    # 3. Calculate cosine similarity
    similarity = pairwise.cosine_similarity(X,y)
    similarity_array = np.array(similarity).flatten()
    df['cosine'] = similarity_array

    # 4. Filter the data with highest Cosine similarity
    person_name, role = '',''
    filter_query = df['cosine'] > threshold
    filtered_data = df[filter_query]
    if len(filtered_data) > 0:
        filtered_data.reset_index(drop = True, inplace= True)
        # argmax returns index of maxium_value
        cosine_argmax = filtered_data['cosine'].argmax()
        person_name, role = filtered_data.loc[cosine_argmax][['Name','Role']]
    else:
        person_name,role='Unknown','Unknown'

    return person_name, role

def name_prediction(image, df):
    image_information = app_sc.get(image)
    image_copy = image.copy()
    # we can use loop to get data of every person and to manipulate the input image 
    # eg draw rectangle around recognized  person's face 
    type(image_information)
    df.head()
    person_name = ""
    for info in image_information:
        bbox = info['bbox'].astype(int)
        embeddings = info['embedding']
        person_name , role = mL_search_algorithm(df,'Facial Features',embeddings)
        x1,y1,x2,y2 = info['bbox'].astype(int)
        # set text_color green if the person name is present else red 
        text_color = (0,255,0) if person_name != 'Unknown' else (0,0,255)
        cv2.rectangle(image_copy,(x1,y1),(x2,y2),text_color,2)
        cv2.putText(image_copy,person_name,(x1,y1-10),cv2.FONT_HERSHEY_DUPLEX,0.5,text_color)
     
    # return image_copy
    return person_names