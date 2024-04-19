# Bibliotecas estándar de Python
import os
import ast
from typing import List

# Manipulación de datos y cálculos numéricos
import numpy as np
import pandas as pd

# Visualización de datos
import matplotlib.pyplot as plt

# Machine learning y análisis de datos
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Trabajo con eBooks
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Variables de entorno
from dotenv import load_dotenv, find_dotenv

# Bibliotecas y módulos relacionados con OpenAI
import openai
from openai import OpenAI

# Funcionalidades específicas de langchain
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate

# Modelos Pydantic para validación y esquemas de datos
from pydantic import BaseModel, Field 

# Importaciones específicas de chromadb (ajusta según sea necesario)
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# Definir constantes

class BookEmbeddingApp:

    def __init__(self, file_path, model_embeddings="text-embedding-3-large", model_llm="gpt-3.5-turbo-0125", api_key_user=None):
        self.file_path = file_path
        self.df_book = None
        self.chroma_collection = None

        if api_key_user:
            self.api_key = api_key_user
        else:
            try:
                _ = load_dotenv(find_dotenv()) # read local .env file
                self.api_key = os.environ['OPENAI_API_KEY']
            except KeyError:
                raise EnvironmentError("API key not found. Please set the OPENAI_API_KEY environment variable or pass it explicitly.")
                
        openai.api_key = self.api_key
        self.openai_client = OpenAI()

        self.model_embeddings = model_embeddings 
        self.model_llm = model_llm
        self.chunk_size=2000 
        self.chunk_overlap=20
        self.csv_path='book_embeddings.csv'  
        self.collection_name='book'
        self.load_from_csv=False
        # Prompts used:

        self.prompt_augment_queries = """ 
                Eres un útil asistente experto en scrum y project management. Tu usuario quiere ampliar conocimientos sobre 
                conceptos muy concretos. Sugiere como máximo 5 enunciados  relacionado con el concepto que busca tu usuario.
                Usa únicamente enunciados cortos, sin oraciones compuestas. Sugiere diversos enunciados con diferentes perspectivas sobre
                el concepto del enunciado.

                Asegurate de que tus enunciados tenga una relación directa con el enunciado original. 

                El resultado tiene que ser un enunciado por línea. NO enumeres las preguntas
                """

        self.prompt_rag = """
                Imagina que eres un asistente virtual especializado en metodologías ágiles, con un enfoque particular en Scrum y la gestión eficaz de proyectos. 
                Tu misión es proporcionar explicaciones simples, directas y precisas que conviertan conceptos complejos en entendimientos claros.
                
                Al interactuar con los usuarios, tu objetivo es desglosar la información de manera que sea accesible para principiantes sin sacrificar 
                la precisión o profundidad necesaria para aquellos más experimentados.
                
                Cuando recibas una solicitud, por favor:
                
                Claridad y Precisión: Usa un lenguaje claro y evita jerga innecesaria. Cuando uses términos técnicos, 
                incluye una breve definición o explicación para garantizar que la información sea comprensible para todos los niveles de experiencia.
                
                Responde Directamente: Asegúrate de que tu respuesta esté directamente relacionada con la consulta del usuario, 
                utilizando únicamente la información proporcionada. Esto garantiza relevancia y utilidad. Tienes prohíbido escribir introducciones o conclusiones. 
                Simplemente desarrolla los conceptos directamente sin añadir información superflua. 
                
                Recuerda, tu papel como asistente virtual no es solo informar, sino también educar e inspirar a aquellos que buscan mejorar 
                sus habilidades en la gestión de proyectos y la implementación de Scrum.

                """


    def init_embedding_function(self):
        '''
        Function that initialize the embedding function
        used by the vector database chromadb. 
        '''
        return embedding_functions.OpenAIEmbeddingFunction(api_key=self.api_key, model_name=self.model_embeddings)


    def extract_text_from_book(self):
        '''
        Function that receives a book in epub format and
        returns the text extracted from the book. 
        '''

        book = epub.read_epub(self.file_path)
    
        text = ""
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.content, 'html.parser')
                text += soup.get_text() + '\n'
        return text


    def split_text(self, raw_book):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                            chunk_overlap=self.chunk_overlap,
                                            ).split_text(raw_book) 


    def create_or_load_df(self):

        if self.csv_path:
            df = pd.read_csv(self.csv_path)
            df['embedding'] = df['embedding'].apply(ast.literal_eval)
        else:
            raw_book = self.extract_text_from_book()
            splits_from_book = self.split_text(raw_book)
            df = pd.DataFrame(splits_from_book, columns=['book_content'])
            df.to_csv(self.csv_path, index=False)
        
        self.df_book=df


    def init_chroma_collection(self, reset_collection=False):

        embedding_function = self.init_embedding_function()
        chroma_client = chromadb.PersistentClient("chromadb_database")

        if reset_collection:
            chroma_client.delete_collection(name=self.collection_name)

        self.chroma_collection = chroma_client.get_or_create_collection(self.collection_name, embedding_function=embedding_function)
        
        if self.chroma_collection.count() == 0:
            self.csv_path = csv_path

            self.create_or_load_df()

            book_docs = self.df_book["book_content"].to_list()
            ids = [str(i) for i in range(len(book_docs))]

            self.chroma_collection.add(ids=ids, documents=book_docs)
            print(f"Total documents in the collection: {self.chroma_collection.count()}")


    def add_new_documents_to_collection(self, documents):
        """
        Añade una lista de nuevos documentos a la colección ChromaDB.
        Esta función asume que `self.chroma_collection` ya está inicializada.

        :param documents: Lista de documentos (texto) para añadir a la colección.
        """
        if not documents:
            print("No hay documentos para añadir.")
            return

        # Generar IDs únicos para los nuevos documentos. Aquí, simplemente se usan índices basados en la cuenta existente.
        # Ajusta esta lógica según sea necesario para asegurar la unicidad.
        start_id = self.chroma_collection.count()
        ids = [str(start_id + i) for i in range(len(documents))]
        
        try:
            self.chroma_collection.add(ids=ids, documents=documents)
            print(f"Se han añadido {len(documents)} nuevos documentos a la colección.")
        except Exception as e:
            print(f"Error al añadir nuevos documentos a la colección: {e}")


    def augment_multiple_query(self, query):
        messages = [
            {
                "role": "system",
                "content": self.prompt_augment_queries
            },
            {"role": "user", "content": query}
        ]

        response = self.openai_client.chat.completions.create(
            model=self.model_llm,
            messages=messages,
        )
        content = response.choices[0].message.content
        content = content.split("\n")
        return content


    def retrieve_documents(self, query):
        # Primero, obtenemos las consultas aumentadas y las combinamos con la consulta original.
        augmented_queries = self.augment_multiple_query(query)
        queries = [query] + augmented_queries

        try:
            # Realizamos una sola consulta a ChromaDB con todas las consultas.
            first_results = self.chroma_collection.query(
                query_texts=queries,
                n_results=10,
                include=['documents']
            )

            # Asumiendo que 'first_results' es una lista de listas de documentos (uno por cada consulta),
            # usamos set para evitar documentos duplicados, mejorando así la eficiencia.
            retrieved_documents = set(doc for docs_list in first_results['documents'] for doc in docs_list)

            # Concatenamos los documentos en un solo string, separados por dos nuevas líneas.
            information = "\n\n".join(retrieved_documents)

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            information = ""

        return information


    def rag(self, query):
        information = self.retrieve_documents(query)

        messages = [
            {
                "role": "system",
                "content": self.prompt_rag 
            },
            {
                "role": "user", 
                "content": f"Concepto: {query}. \n Informacion: {information}"
            }
            ]
            
        response = self.openai_client.chat.completions.create(
            model=self.model_llm,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content, information

class TextProcessingSystem:
    """
    Sistema integrado para el procesamiento de texto que incluye lectura de archivos Markdown,
    división de textos en fragmentos, generación de texto utilizando modelos de lenguaje y
    clustering basado en embeddings de texto.
    """

    def __init__(self, markdown_file_path=None, model_embeddings="text-embedding-3-large", model_llm='gpt-3.5-turbo-0125', api_key_user=None):
        self.markdown_file_path = markdown_file_path

        self.model_embeddings = model_embeddings

        if api_key_user:
            self.api_key = api_key_user
        else:
            try:
                _ = load_dotenv(find_dotenv()) # read local .env file
                self.api_key = os.environ['OPENAI_API_KEY']
            except KeyError:
                raise EnvironmentError("API key not found. Please set the OPENAI_API_KEY environment variable or pass it explicitly.")
                
        openai.api_key = self.api_key
        
        self.model = ChatOpenAI(api_key=self.api_key, temperature=0, model=model_llm)
        
        self.embeddings = OpenAIEmbeddings(model=self.model_embeddings)
        self.num_clusters = None

        self.chunk_size = 1000
        self.chunk_overlap=0
        self.csv_path='csv_with_embeddings.csv'
        self.question_csv_path="questions.csv"


        # Este prompt recibe un grupo de textos y los tiene que unir para hacer preguntas.
        self.prompt_map_template_preguntas = '''Eres un asistente que ayuda a extraer información relevante de extractos de texto. Analiza los extractos proporcionados, y elabora un resumen que describa claramente y de forma directa las ideas generales del conjunto de los extractos. El resumen tiene que ser escrito de forma directa como si fuera en primera persona. Sin introducciones ni nada parecido, simplemente describe los conceptos a su forma más básica. Solo puedes usar ESTRICA Y ÚNICAMENTE la información proporcionada en los extractos propocionados.
        Extractos:
        {questions}'''

        
        # Este prompt recibe todas las preguntas, y tiene que filtrar las que estén repetidas.
        self.prompt_questions_list = ''' Vas a recibir una lista de python con una serie de enunciados con 
                    diferentes conceptos que describen el contenido de un libro concreto.

                    Tu tarea es analizar todos los enunciados y reescribir en una lista de python los conceptos
                    únicos del conjunto de todos los enunciados que recibes. Cuando un enunciado sea demasiado 
                    ambiguo, reformulalos para que sean más claros.

                    El objetivo es usar esos enunciados para describir claramente el contenido del libro en concreto.
                     \n{format_instructions}\n{query}\n'''
      
    def read_markdown_file(self):
        """
        Lee un archivo Markdown y lo divide en secciones basadas en los encabezados.
        """
        with open(self.markdown_file_path, 'r') as file:
            markdown_content = file.read()

        headers_to_split_on = [
            ("#", "Header 1"), 
            ("##", "Header 2"), 
            ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        self.header_splits = markdown_splitter.split_text(markdown_content)

    def get_embedding_openAI(self, text):
        """
        Obtiene el embedding de un texto utilizando OpenAI.
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Error  el texto:\n{text} \n error: {e}")
            return np.zeros(512)

    def create_dataframe(self):
        """
        Divide el texto dado en fragmentos más pequeños basándose en el tamaño del fragmento y la superposición especificados.
        """
        self.read_markdown_file()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        documents = text_splitter.split_documents(self.header_splits)

        page_content = [doc.page_content for doc in documents]

        self.df_documents = pd.DataFrame({'page_content': page_content})
        self.df_documents['embedding'] = self.df_documents['page_content'].apply(self.get_embedding_openAI)

    def get_dataframe(self, load=False, ):
        if load:
            self.df_documents = pd.read_csv(self.csv_path)
            self.df_documents['embedding'] = self.df_documents['embedding'].apply(ast.literal_eval)

        else:
            self.create_dataframe()
            self.df_documents.to_csv(self.csv_path)

    def optimal_cluster(self):
        self.embeddingMatrix = np.array(self.df_documents["embedding"].tolist())
        

        n_samples = self.embeddingMatrix.shape[0]
        max_possible_clusters = n_samples - 1
        cluster_range = range(2, min(50, max_possible_clusters))

        silhouette_scores = []

        # Convertir el dataframe a un formato adecuado para clustering


        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
            kmeans.fit(self.embeddingMatrix)
            labels = kmeans.labels_
            score = silhouette_score(self.embeddingMatrix, labels)
            silhouette_scores.append(score)

        best_n_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
        self.num_clusters = best_n_clusters
        

        # Crea y devuelve la figura
        fig, ax = plt.subplots()
        ax.plot(cluster_range, silhouette_scores, marker="o")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score for Optimal Cluster Count")
        
        return fig, ax # Devuelve el objeto figura

    def create_clusters(self):
        """
        Realiza clustering en los documentos basándose en embeddings de texto.
        """
        if not self.num_clusters:
            self.optimal_cluster()

        # Realizar el clustering con el número óptimo de clusters
        kmeans = KMeans(n_clusters=self.num_clusters, init="k-means++", random_state=42)
        labels = kmeans.fit_predict(self.embeddingMatrix)
        self.df_documents["Cluster"] = labels
        
        # sort clusters by their size, and create a new dataframe with the size and cluster number in sorted order. Maintain other columns
        self.size_and_cluster_number = pd.DataFrame(self.df_documents.groupby("Cluster").size().sort_values(ascending=False))

    def create_doc(self, messages, max_tokens=8000):
        input_doc = '\n\n'.join(messages)

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=max_tokens,chunk_overlap=30,separator="\n\n")
        
        # Sanity check
        split_texts = text_splitter.split_text(input_doc)
        
        return split_texts[0]
    
    def grouping_text(self):
        self.create_clusters()
        
        docs = []  # Lista para almacenar los documentos
        
        # Itera a través de cada cluster y sus tamaños (aunque el tamaño no se usa directamente aquí)
        for cluster_number, size in self.size_and_cluster_number.iterrows():
            messages = self.df_documents[self.df_documents.Cluster == cluster_number].page_content.values
            doc = self.create_doc(messages)  # Asume que esta función devuelve un único string o documento
            docs.append(doc)
        
        # Crea un DataFrame temporal con los nuevos datos
        new_docs_df = pd.DataFrame({'doc': docs})
        
        # Verifica si self.docs ya existe
        if hasattr(self, 'docs') and not self.docs.empty:
            # Si self.docs ya existe y contiene datos, añade los nuevos documentos a él
            self.docs = self.docs.append(new_docs_df, ignore_index=True)
        else:
            # Si self.docs no existe o está vacío, inicialízalo con los nuevos documentos
            self.docs = new_docs_df

    def run_map_questions(self, input_doc):
        prompt = PromptTemplate(
            template=self.prompt_map_template_preguntas,
            input_variables=["questions"], 
            )
        output_parser = StrOutputParser()
        
        chain = prompt | self.model | output_parser
        
        return chain.invoke({"questions": input_doc})

    def create_questions(self, grouping_text_activated=False, num_clusters=None, load_questions=False):
        if not load_questions:
            if not grouping_text_activated:
                self.num_clusters = num_clusters
                self.grouping_text()
            # Aplica 'self.run_map_questions' a cada documento en la columna 'doc' y almacena los resultados en una nueva columna 'questions'
            self.docs['questions'] = self.docs['doc'].apply(self.run_map_questions)

            self.docs.to_csv(self.question_csv_path)

        else:
            self.docs = pd.read_csv(self.question_csv_path)
        
        return self.docs

    def generate_list_of_questions(self, filename='list_questions.txt' ):
        ''' 
        Necesita antes que se inicie grouping_text()
        '''

        questions = self.docs['questions'].tolist()

        class Questions(BaseModel):
            values: List[str] = Field(description='Lista de los enunciados más importantes')

        parser = PydanticOutputParser(pydantic_object=Questions)

        prompt = PromptTemplate(
            template=self.prompt_questions_list,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.model | parser

        self.response = chain.invoke({"query": questions})
        
        with open(filename, 'w', encoding='utf-8') as f:
            for item in self.response.values:
                f.write(item + '\n')

