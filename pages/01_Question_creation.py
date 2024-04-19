import streamlit as st
from modules.ai_utils import TextProcessingSystem 
import pandas as pd


# Función auxiliar para manejar el cambio en las preguntas
def rewrite_question(i):
    st.session_state.df_queries.at[i, 'questions'] = st.session_state[f'question_{i}']

def save_dataframe():
    st.session_state.df_queries.to_csv(st.session_state.dataframe_csv_path)



# Lógica para manejar el área de texto y la reescritura de preguntas
if ('df_queries' in st.session_state) and st.session_state.df_queries is not None:

    # Botón para guardar el dataframe
    st.button('Save Changes', key='saveDataframe', on_click=save_dataframe)

    for i, row in st.session_state.df_queries.iterrows():
        st.text_area(f"question {i+1}", 
                     value=row.questions,
                     key=f"question_{i}",
                     height=200,
                     on_change=rewrite_question,
                     args=(i,))
        with st.expander("See extracts"):
            st.text_area(value=row.doc, 
                         label=f"docs {i+1}",
                         height=500)

else:
    st.write("Generate Questions First!")

