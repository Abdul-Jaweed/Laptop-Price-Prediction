import streamlit as st
import pickle
import numpy as np

def space():
    st.markdown("<br>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Hello",
    page_icon="😎",
)


st.write("# Heya Everyone ! 👋")

st.markdown("# :red[Abdul Jaweed]")
st.subheader(" :blue[Data Scientist]")
space()


st.markdown(
    """
    #### I specialize in Data Science and Machine Learning and my passion is to take a cup of coffee and start finding some meaningful insight in messy data and build some great model...
    """
)

space()

st.write("#### Connect me on [LinkedIn](https://www.linkedin.com/in/abdul-jaweed-datascientist/)")
st.write("#### Follow me on [GitHub](https://github.com/Abdul-Jaweed)")
st.write("#### Follow me on [Medium](https://medium.com/@abduljaweed)")