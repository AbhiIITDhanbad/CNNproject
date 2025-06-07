# import os
# import pickle
# actors=os.listdir('Actors_Data')
# print(actors)
# filenames=[]
# for actor in actors:
#     for file in os.listdir(os.path.join('Actors_Data',actor)):
#         filenames.append(os.path.join('Actors_Data',actor,file))
#
# # print(filenames)
# pickle.dump(filenames,open('filenames.pkl','wb'))
# # print(len(filenames))
# from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFace
from tqdm import tqdm
import numpy as np
import pickle
filenames=pickle.load(open('filenames.pkl','rb'))
model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
# print(model.summary())
def feature_extractor(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img)
    result=model.predict(preprocessed_img).flatten()
    return result

features=[]
for file in tqdm(filenames):
    features.append(feature_extractor(file,model))
pickle.dump(features,open('embedding.pkl','wb'))








