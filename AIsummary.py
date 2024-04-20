import streamlit as st
from modules.ai_utils import TextProcessingSystem, BookEmbeddingApp
import pandas as pd
import logging

def change_embedding_status_toggle():
    # Actualizar el estado de la sesión cuando el toggle cambie
    if st.session_state.embeddings_exist_toggle:
        st.session_state['embeddings_exist'] = True
    else:
        st.session_state['embeddings_exist'] = False
    
def change_question_status_toggle():
    # Actualizar el estado de la sesión cuando el toggle cambie
    if st.session_state.questions_exist_toggle:
        st.session_state['questions_exist'] = True
    else:
        st.session_state['questions_exist'] = False

def generate_questions():
    if "Text_Process" not in st.session_state:
        try:
            st.session_state.Text_Process = TextProcessingSystem(markdown_file_path=st.session_state.markdown_file_path, api_key_user=st.session_state.api_key)
        except EnvironmentError as e:
            st.error(f'Failed to initialize app: {str(e)}')

    st.session_state.Text_Process.get_dataframe(load=st.session_state.embeddings_exist)
    st.session_state.df_queries = st.session_state.Text_Process.create_questions(load_questions=st.session_state.questions_exist)
    
    
    st.session_state.df_queries['response'] = None
    st.session_state.df_queries['information'] = None
    
    st.success("Sucess!")

def upload_dataframe():
    st.session_state.df_queries = pd.read_csv(st.session_state.dataframe_csv_path)

def initiate_generator():
    try:
        logging.info(f"Initializing Text Processing System with: {st.session_state.markdown_file_path}, {st.session_state.api_key}")
        st.session_state.Text_Process = TextProcessingSystem(
            markdown_file_path=st.session_state.markdown_file_path, 
            api_key_user=st.session_state.api_key
        )
    except Exception as e:
        logging.error("Failed to initialize Text Processing System", exc_info=True)
        raise e 
def rewrite_prompt():
    st.session_state.Text_Process.prompt_map_template_preguntas = st.session_state.new_prompt

def update_object():
    keys = list(st.session_state.questions_parameters.keys())
    # Acceder a los datos editados desde el widget data_editor
    for edited_data in st.session_state['extracts_object_modification_editor']['edited_rows']:
        new_value = st.session_state['extracts_object_modification_editor']['edited_rows'][edited_data]['value']
        st.session_state.questions_parameters[keys[edited_data]]=new_value


def update_rag_object():
    keys = list(st.session_state.rag_parameters.keys())
    # Acceder a los datos editados desde el widget data_editor
    for edited_data in st.session_state['rag_modification_editor']['edited_rows']:
        new_value = st.session_state['rag_modification_editor']['edited_rows'][edited_data]['value']
        st.session_state.rag_parameters[keys[edited_data]]=new_value

def apply_rag_changes():
    keys = list(st.session_state.rag_parameters.keys())

    for key in keys:
        current_value = getattr(st.session_state.rag, key)
        if current_value != st.session_state.rag_parameters[key]:
            st.info(f'Cambiando {current_value} to {st.session_state.rag_parameters[key]}')
            setattr(st.session_state.rag, key, st.session_state.rag_parameters[key])

def update_api_key():
    st.session_state.api_key = st.session_state.api_key_input

def activate_rag_function():
    if 'rag' not in st.session_state:
        st.session_state['rag'] = BookEmbeddingApp(file_path=st.session_state.book_path, api_key_user=st.session_state.api_key, csv_path_load=False)
        st.session_state.rag.init_chroma_collection()

if not 'dataframe_csv_path' in st.session_state:
    st.session_state.dataframe_csv_path = 'dataframe_text.csv'

if "df_queries" not in st.session_state:
    st.session_state['df_queries'] = None

# Inicializar el valor en el estado de la sesión si aún no existe
if 'embeddings_exist' not in st.session_state:
    st.session_state['embeddings_exist'] = True  # Asume un valor por defecto

if 'questions_exist' not in st.session_state:
    st.session_state['questions_exist'] = True  # Asume un valor por defecto

if 'markdown_file_path' not in st.session_state:
    st.session_state['markdown_file_path'] = 'resumen_libro.md'

if 'questions_parameters' not in st.session_state:
    st.session_state['questions_parameters'] = {
        'model_embeddings': 'text-embbedding-3-large',
        'model_llm': 'gpt-3.5-turbo-0125',
        'chunk_size': 1000,
        'chunk_overlap': 0,
        'csv_path':'csv_with_embeddings.csv',
        'question_csv_path':"questions.csv",
        'prompt_map_template_preguntas': '''Eres un asistente que ayuda a extraer información relevante de extractos de texto. \n\nAnaliza los extractos proporcionados, y elabora un resumen que describa claramente y de forma directa las ideas generales del conjunto de los extractos. \n\nEl resumen tiene que ser escrito de forma directa como si fuera en primera persona. Sin introducciones ni nada parecido, simplemente describe los conceptos a su forma más básica. \n\nSolo puedes usar ESTRICA Y ÚNICAMENTE la información proporcionada en los extractos propocionados.
        Extractos:
        {questions}'''
    }

if 'book_path' not in st.session_state:
    st.session_state['book_path'] = "scrum_sutherland.epub"

if 'rag_parameters' not in st.session_state:
    st.session_state['rag_parameters'] = {
        'model_embeddings': 'text-embedding-3-large',
        'model_llm': "gpt-3.5-turbo-0125",
        'chunk_size': 2000,
        'chunk_overlap': 20,
        'csv_path': 'book_embeddings.csv',
        'collection_name': 'book',
        'load_from_csv': False
    }

if 'api_key' not in st.session_state:
    st.session_state['api_key']=None


st.header("Importing Book to Your Second Brain with a RAG")

st.text_input('OpenAI API Key:',   
    key='api_key_input',
    type='password',
    on_change=update_api_key
    )

st.subheader('Extract Configuration')

markdown_path = st.text_input(placeholder=st.session_state.markdown_file_path, 
        label='Documento Markdown:',
        value=st.session_state.markdown_file_path)
st.session_state['markdown_file_path'] = markdown_path

col1, col2, col3 = st.columns(3)

with col1:
    initate_question_generator = st.toggle('Initiate Question Generator', 
                                    key='initiate_question_generator_toggle',
                                    on_change=initiate_generator)

with col2:
    embedding_dataframe = st.toggle('Embeddings exist?', 
                                    value=st.session_state['embeddings_exist'], 
                                    key='embeddings_exist_toggle',
                                    on_change=change_embedding_status_toggle)

with col3:
    questions_exist_toggle = st.toggle('Questions exist?', 
                                    value=st.session_state['questions_exist'], 
                                    key='questions_exist_toggle',
                                    on_change=change_question_status_toggle)

col_generate, col_upload = st.columns(2)

with col_generate:
    # Botón para generar preguntas
    st.button('Generate Questions: ', 
            key='generateQuestion', 
            on_click=generate_questions)

with col_upload:
    # Botón para cargar el dataframe
    st.button('Upload Dataframe', 
            key='uploadDataframe', 
            on_click=upload_dataframe)


with st.expander('Prompt Template'):
    st.text_area('Template',
            value=st.session_state.questions_parameters['prompt_map_template_preguntas'],
            on_change=rewrite_prompt,
            key='new_prompt',
            height=400)

st.data_editor(st.session_state.questions_parameters,
on_change=update_object,
key='extracts_object_modification_editor',
)





st.subheader('Rag Configuration')

col_book_path, col_rag = st.columns(2)
with col_book_path:
    book_path = st.text_input(placeholder=st.session_state.book_path, 
            label='Libro para hacer el rag:',
            value=st.session_state.book_path)
    st.session_state['book_path'] = book_path

    st.toggle("Activate RAG", 
              key="toggle_rag",
              on_change=activate_rag_function)
 

with col_rag:
    st.data_editor(st.session_state.rag_parameters,
        on_change=update_rag_object,
        key='rag_modification_editor',
    )

    st.button('Apply changes',
            key='apply_changes_button',
            on_click=apply_rag_changes)

