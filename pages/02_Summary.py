import streamlit as st
from modules.ai_utils import BookEmbeddingApp 
import pandas as pd

def dataframe_to_markdown():
    """
    Convierte un DataFrame a un documento de Markdown.
    
    Args:
    - df (pd.DataFrame): DataFrame que contiene las columnas 'questions' y 'response'.
    - csv_path (str): La ruta del archivo CSV para usar en el encabezado principal.
    
    Returns:
    - str: Documento de Markdown generado.
    """

    df = st.session_state.df_queries
    csv_path = st.session_state.dataframe_csv_path

    # Quita la extensión .csv del nombre del archivo para usarlo como encabezado principal
    header = csv_path.split('.csv')[0]

    markdown_path = header + ".md"
    markdown_doc = f"# {header}\n\n"
    
    # Itera sobre cada fila del DataFrame
    for i, row in df.iterrows():
        if not pd.isna(row['response']) and row['response'].strip():
            markdown_doc += f"## {row['questions']}\n\n{row['response']}\n\n"
    
    # Escribe el documento de Markdown en un archivo
    with open(markdown_path, 'w', encoding='utf-8') as file:
        file.write(markdown_doc)

    return markdown_doc


def generate_response(index):
    query = st.session_state.df_queries.at[index, 'questions']

    if st.session_state.activate_rag:
        response, information = st.session_state.rag.rag(query)

        st.session_state.df_queries.at[index, 'response'] = response
        st.session_state.df_queries.at[index, 'information'] = information


def rewrite_questions(i):
    st.session_state.df_queries.at[i, 'questions'] = st.session_state[f'questions_{i}']


def rewrite_response(i):
    st.session_state.df_queries.at[i, 'response'] = st.session_state[f'response_{i}']


# Función para añadir un nuevo questions
def add_questions(add_new_questions):
    # Crea un nuevo DataFrame para el questions a añadir
    new_row = pd.DataFrame({'questions': [add_new_questions], 'response': ['']})
    # Concatena el nuevo DataFrame con el existente en st.session_state
    st.session_state.df_queries = pd.concat([new_row, st.session_state.df_queries], ignore_index=True)
        # Actualizar session_state para reflejar los cambios
    st.session_state['df_queries'] = st.session_state.df_queries


def update_prompt_queries():
    st.session_state.rag.prompt_augment_queries = st.session_state.prompt_queries

def update_prompt_rag():
    st.session_state.rag.prompt_rag = st.session_state.prompt_rag


if not 'dataframe_csv_path' in st.session_state:
    st.session_state.dataframe_csv_path = 'dataframe_text.csv'

if 'import_option' not in st.session_state:
    st.session_state.import_option = None

if 'rag' in st.session_state:
    col_save, col_markd = st.columns(2)
    with col_save:            
        if st.button('Save Changes', key='save_button'):
            try:
                st.session_state.df_queries.to_csv(st.session_state.dataframe_csv_path, index=False)
                st.write(f"Saved in { st.session_state.dataframe_csv_path}")
            except Exception as e:
                print("Error: ", e)
    with col_markd:
        if st.button("Save Markdown", key="save_markdown"):
            try:
                dataframe_to_markdown()
            except Exception as e:
                print("Error ", e)

    with st.expander('Prompt Query Expander'):
        st.text_area('Prompt para mejorar los queries',
            value=st.session_state.rag.prompt_augment_queries,
            key='prompt_queries',
            on_change=update_prompt_queries,
            height=400
        )

    with st.expander('Prompt RAG'):
        st.text_area('Prompt para responder preguntas',
            value = st.session_state.rag.prompt_rag,
            key='prompt_rag',
            on_change=update_prompt_rag,
            height=400
        )


    with st.form("New questions", clear_on_submit=True):
        question=st.text_input('New questions:')
        submitted = st.form_submit_button("Add")
        if submitted:
            add_questions(question)
            st.rerun()


    # Mostrar y editar questionss existentes
    for i, row in st.session_state.df_queries.iterrows():
        with st.container():
            # Editar el questions
            questions = st.session_state.df_queries.at[i, 'questions']
            response = st.session_state.df_queries.at[i, 'response']
            information = st.session_state.df_queries.at[i, 'information']


            new_questions = st.text_area(f"questions {i+1}", 
                                        value=questions, 
                                        key=f'questions_{i}', 
                                        height=200,
                                        on_change=rewrite_questions, 
                                        args=(i, ))

            if st.button('Generate Response', key=f'generate_response_{i}'):
                generate_response(i)
                response = st.session_state.df_queries.at[i, 'response']
                information = st.session_state.df_queries.at[i, 'information']

            if not pd.isna(response):
                if response:   
                    response_text = st.text_area(f"Response {i+1}", 
                                                    value=response,
                                                    height=400, 
                                                    key=f'response_{i}',
                                                    on_change=rewrite_response,
                                                    args=(i, ))
            
            if not pd.isna(information):
                if information:
                    with st.expander('See extracts'):
                        st.text_area(f"Information {i+1}",
                                                        value=information,
                                                        height=400, 
                                                        key=f'information_{i}',)
            
        st.divider()

else:
    st.write('You must initiate the rag first!')

        


