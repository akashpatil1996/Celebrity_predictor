import streamlit as st
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace


feature_list = pickle.load(open('features.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))
detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

st.title('Which Celebrity do you look like?')

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    if not results:
        st.error("No faces detected in the image. Please upload another image")

    x, y, width, height = results[0]['box']
    face = img[y:y+height, x:x+width]
    image = Image.fromarray(face)
    image = image.resize((224,224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded = np.expand_dims(face_array,axis=0)
    preprocessed = preprocess_input(expanded)
    res = model.predict(preprocessed).flatten()
    return res

def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


uploaded_image = st.file_uploader('Upload you photo')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_img = Image.open(uploaded_image)
        features = extract_features(os.path.join('uploads',uploaded_image.name), model, detector)
        index_pos = recommend(feature_list,features)
        pred_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        st.success('You look like '+ pred_actor)
        col1, col2 =st.columns(2)
        with col1:
            st.text('Your image')
            st.image(display_img,width=200)
        with col2:
            t = st.text(pred_actor)
            st.image(filenames[index_pos], width=200)


