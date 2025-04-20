import chromadb
import os
import logging

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Carga y Preparación de Datos desde Archivo ---
data_file = "saber_pro_data.txt"
saber_pro_docs = []
saber_pro_ids = []

logger.info(f"Intentando leer datos desde '{data_file}'...")
try:
    with open(data_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Dividir el texto en bloques. Asume que cada bloque Q&A está separado
    # por exactamente UNA línea en blanco (\n\n).
    # .strip() elimina espacios/líneas en blanco al inicio/final del archivo completo.
    raw_docs = full_text.strip().split('\n\n')

    # Limpiar cada documento y generar IDs
    for i, doc in enumerate(raw_docs):
        cleaned_doc = doc.strip()
        if cleaned_doc: # Solo añadir si el bloque no está vacío después de limpiar
            saber_pro_docs.append(cleaned_doc)
            # Generar IDs simples basados en el índice
            saber_pro_ids.append(f"saberpro_faq_{i+1}")

    if not saber_pro_docs:
        logger.error(f"No se encontraron documentos válidos en '{data_file}'. ¿Está vacío o mal formateado?")
        exit()
    else:
         logger.info(f"Se cargaron y procesaron {len(saber_pro_docs)} documentos desde '{data_file}'.")

except FileNotFoundError:
    logger.error(f"Error CRÍTICO: El archivo '{data_file}' no se encontró.")
    logger.error("Asegúrate de que el archivo exista en el mismo directorio que este script y tenga el contenido formateado.")
    exit()
except Exception as e:
    logger.error(f"Error inesperado al leer o procesar el archivo '{data_file}': {e}", exc_info=True)
    exit()

# --- 2. Configuración de ChromaDB ---
# Cliente PERSISTENTE para que los datos no se pierdan.
# Misma ruta usada en main_rag.py
db_directory = "chroma_db_saberpro"
db_path = os.path.abspath(db_directory)
logger.info(f"Usando cliente ChromaDB persistente en: {db_path}")

if not os.path.exists(db_path):
    try:
        os.makedirs(db_path)
        logger.info(f"Directorio para ChromaDB creado en: {db_path}")
    except OSError as e:
        logger.error(f"Error CRÍTICO al crear el directorio para ChromaDB en {db_path}: {e}")
        exit()

try:
    client = chromadb.PersistentClient(path=db_path)
except Exception as e:
    logger.error(f"Error CRÍTICO al inicializar PersistentClient de ChromaDB en {db_path}: {e}", exc_info=True)
    exit()

# Nombre para la colección (tiene q coincidir con main_rag.py)
collection_name = "preguntas_frecuentes_saberpro"

# Crear la colección (o recuperarla si existe)
logger.info(f"Intentando obtener o crear la colección: '{collection_name}'")
try:
    collection = client.get_or_create_collection(name=collection_name)
    logger.info(f"Colección '{collection_name}' lista.")
except Exception as e:
     logger.error(f"Error al obtener/crear la colección ChromaDB '{collection_name}': {e}", exc_info=True)
     exit()


# --- 3. Añadir Documentos a la Colección (con verificación de existencia) ---
try:
    # Intentar obtener todos los documentos existentes para comparar IDs.
    existing_data = collection.get(ids=saber_pro_ids) # Pide los IDs que intentamos añadir
    existing_ids = set(existing_data['ids'])
    logger.info(f"Se encontraron {len(existing_ids)} IDs de la lista actual que ya existen en la colección.")
except Exception:
    # Si get() falla (por ejemplo la colección está vacía o los IDs no existen), se asume que no hay IDs existentes.
    logger.info("No se encontraron IDs existentes (o la colección está vacía). Se intentará añadir todos los documentos leídos.")
    existing_ids = set()


new_docs_to_add = []
new_ids_to_add = []
for i, doc_id in enumerate(saber_pro_ids):
    if doc_id not in existing_ids:
        new_docs_to_add.append(saber_pro_docs[i])
        new_ids_to_add.append(doc_id)

if new_ids_to_add:
    logger.info(f"Añadiendo {len(new_ids_to_add)} nuevos documentos a la colección '{collection_name}'...")
    try:
        collection.add(
            documents=new_docs_to_add,
            ids=new_ids_to_add
        )
        logger.info("Nuevos documentos añadidos con éxito.")
    except Exception as e:
        logger.error(f"Error al añadir documentos a la colección '{collection_name}': {e}", exc_info=True)
else:
    logger.info(f"No se añadieron nuevos documentos. Todos los IDs del archivo '{data_file}' ya existen en la colección '{collection_name}'.")

# Verificar el conteo final
try:
    final_count = collection.count()
    logger.info(f"La colección '{collection_name}' ahora contiene {final_count} documentos.")
except Exception as e:
    logger.error(f"No se pudo verificar el conteo final de la colección: {e}")


# --- 4. Consultar la Colección (Ejemplo) ---
# Realizar una consulta de ejemplo para verificar que funciona
query_text = ["¿Qué hago si olvidé mi contraseña de PRISMA?"]

logger.info(f"\nRealizando consulta de ejemplo con: '{query_text[0]}'")
try:
    results = collection.query(
        query_texts=query_text,
        n_results=3 
    )

    logger.info("\n--- Resultados de la Consulta de Ejemplo ---")
    if results and results.get('documents') and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            logger.info(f"Resultado {i+1}:")
            # Imprime solo una parte del documento
            preview = (doc[:250] + '...') if len(doc) > 250 else doc
            logger.info(f"  Texto (preview): {preview.replace(os.linesep, ' ')}") # Reemplaza saltos de línea para preview
            # logger.info(f"  Distancia: {results['distances'][0][i]}")
            # logger.info(f"  ID: {results['ids'][0][i]}")
            logger.info("-" * 20)
    else:
        logger.info("No se encontraron resultados para la consulta de ejemplo.")
except Exception as e:
    logger.error(f"Error durante la consulta de ejemplo: {e}", exc_info=True)

logger.info("\nScript de carga/verificación de ChromaDB completado.")