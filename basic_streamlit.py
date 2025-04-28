import streamlit as st
from PIL import Image
<<<<<<< HEAD
import sst_2_logistic_regression
img = Image.open("code_sst_2.png")
=======
img = Image.open("code_sst_2.png")
import sst_2_logistic_regression
>>>>>>> 5858796 (HII FIRST COMMIT FROM VSCODE)


st.title("Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank")

st.header("Training our model using SVM")

if st.button("classification report using SVM"):
    st.image(img, width = 2000)

name = st.text_input("Enter ur review about bahubali", "Type Here ...")

# display the name when the submit button is clicked
# .title() is used to get the input text string
if(st.button('Submit')):
    out = sst_2_logistic_regression.predict_sentiment(name)
    st.text(out)
