import streamlit as st
from fastai.vision.all import *
import plotly.express as px
#title

st.title('Classification the transport')

#uploading the picture

file = st.file_uploader('Upload picture', type = ['png', 'gif', 'svg', 'jpeg'])
if file:

    st.image(file)
    #PIL image
    img = PILImage.create(file)

    #model
    model = load_learner('/Users/bahodiralayorov/Desktop/PYTHON_SCRIPTS/transport/transport_model (1).pkl')

    #prediction

    pred, pred_id, prob = model.predict(img)

    st.success(f"Prediction: {pred}")
    st.info(f'Eccuracy: {prob[pred_id]*100:.1f}%')


    #plotting

    fig = px.bar(x=prob*100, y = model.dls.vocab)
    st.plotly_chart(fig)