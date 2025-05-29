from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from google.generativeai import GenerativeModel, configure, types
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK
from starlette.websockets import WebSocketState
# para solucionar el error en azure respecto a sqlite3 en chromadb
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("SQLite version successfully overridden by pysqlite3.") # Log para confirmar
except ImportError:
    print("pysqlite3 not found, using system's sqlite3. This might cause issues with ChromaDB.")
    pass
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import json
import uvicorn
import logging
import os
from dotenv import load_dotenv

# --- Configuración de Logging ---
LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración del LLM ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("No se encontró la variable de entorno GOOGLE_API_KEY")
    exit()

# Configurar Gemini
configure(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.0-flash"

# --- Configuración de ChromaDB (DB Vectorial) ---
CHROMA_DB_FOLDER_IN_REPO = "chroma_db_saberpro"
db_path = os.path.abspath(CHROMA_DB_FOLDER_IN_REPO)

logger.info(f"Intentando acceder a la base de datos ChromaDB pre-construida desde el repositorio en: {db_path}")

if not os.path.exists(db_path):
    logger.error(f"Error CRÍTICO: El directorio de la base de datos '{db_path}' NO se encontró. "
                 f"Asegúrate de que la carpeta '{CHROMA_DB_FOLDER_IN_REPO}' con la DB pre-construida "
                 "esté en la raíz de tu repositorio GitHub y se haya clonado correctamente.")
    exit()

if not os.access(db_path, os.R_OK):
    logger.error(f"Error CRÍTICO: No hay permisos de lectura para el directorio de la base de datos '{db_path}'.")
    exit()

logger.info(f"Directorio de la base de datos ChromaDB '{db_path}' encontrado y legible.")

# Inicializar la función de embedding (debe ser la misma que se usó para crear la DB)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-base")
logger.info(f"Usando embedding function en retrieval: {embedding_fn.model_name}")

# Inicializar el cliente ChromaDB para leer la base de datos existente
logger.info(f"Intentando inicializar cliente ChromaDB persistente para leer desde: {db_path}")
db_client = None
collection = None
try:
    db_client = chromadb.PersistentClient(path=db_path)
    logger.info(f"Cliente ChromaDB inicializado correctamente desde '{db_path}'.")

    collection_name = "preguntas_frecuentes_saberpro"

    collection = db_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    logger.info(f"Conectado a la colección ChromaDB existente: '{collection_name}'")

    count = collection.count()
    logger.info(f"La colección '{collection_name}' contiene {count} documentos.")
    if count == 0:
        logger.warning(f"ADVERTENCIA: La colección ChromaDB '{collection_name}' está vacía. "
                       "Verifica que la base de datos en el repositorio esté completa y no corrupta.")
    else:
        logger.info("Base de datos de conocimiento (ChromaDB) cargada y lista desde el repositorio.")

except chromadb.errors.CollectionNotDefinedError:
    logger.error(f"Error CRÍTICO: La colección '{collection_name}' no se encontró en la base de datos en '{db_path}'. "
                 "Asegúrate de que la base de datos subida al repositorio esté completa y contenga esta colección.")
    exit()
except Exception as e:
    logger.error(f"Error CRÍTICO durante la inicialización de ChromaDB o conexión a la colección '{collection_name}': {e}", exc_info=True)
    logger.error("La aplicación no podrá funcionar correctamente sin la base de datos vectorial.")
    exit()

# --- System Prompt (Instrucciones para el LLM) ---
system_prompt = """
Eres un asistente virtual experto en responder preguntas sobre las pruebas Saber Pro en Colombia, basándote estricta y únicamente en la información proporcionada en el contexto recuperado para cada pregunta. Sigue estas instrucciones con máxima precisión:
1.  Tu única fuente de conocimiento es el texto proporcionado bajo el título 'Contexto'. No uses información externa ni hagas suposiciones.
2.  Analiza la pregunta del usuario y cada fragmento del contexto proporcionado.
3.  Responde solo con información que esté directamente relacionada con la pregunta específica del usuario y que se encuentre explícitamente en el contexto.
4.  Si el contexto contiene varios fragmentos o detalles (como enlaces, fechas, correos), asegúrate de usar solamente aquellos detalles que pertenecen a la respuesta de la pregunta actual del usuario. No mezcles información de diferentes preguntas o temas presentes en el contexto si no son relevantes para la consulta específica. Presta atención a qué fragmento de contexto responde qué.
5.  Si el contexto recuperado describe pasos a seguir o instrucciones detalladas, asegúrate de incluir todos esos pasos en tu respuesta de manera clara y ordenada.
6.  Sé claro, directo, formal y servicial. Evita frases introductorias largas.
7.  No menciones que eres un modelo de lenguaje, ia o un bot. Eres un asistente para consultas de Saber Pro.
8.  Asegúrate de que las url en el contexto estén completas y sean accesibles. Si el contexto menciona un enlace, asegúrate de que esté bien formateado y sea funcional.
9.  Asegúrate de dar respuestas completas y autoconclusivas. No dejes respuestas a medias o incompletas.
10. Si la pregunta del usuario no está relacionada con las pruebas Saber Pro o con temas oficiales del ICFES, responde educadamente que esa consulta no pertenece a tu área de especialización y que estás diseñado únicamente para brindar asistencia sobre los exámenes Saber Pro del ICFES.
11. Si el contexto contiene un enlace indicado dentro de paréntesis y precedido por el texto exacto Link:, extrae y devuelve únicamente la URL completa que aparece inmediatamente después de Link: dentro del paréntesis, sin incluir el texto Link: ni los paréntesis.
12. Si el contexto menciona un enlace, asegúrate de que esté bien formateado y sea funcional.

A continuación se presenta el contexto relevante recuperado de la base de datos para la consulta actual:
"""

# --- Cliente Gemini ---
try:
    model = GenerativeModel(MODEL)
    logger.info(f"Cliente Gemini configurado con modelo: {MODEL}")
except Exception as e:
    logger.error(f"Error al configurar el cliente Gemini: {e}", exc_info=True)
    exit()

# --- Configuración de FastAPI (Server Web) ---
app = FastAPI(title="Saber Pro RAG Chatbot API")

# --- Servir archivos estáticos ---
static_dir = "static"
try:
    if not os.path.exists(static_dir):
        logger.warning(f"El directorio estático '{static_dir}' no existe. Creándolo.")
        os.makedirs(static_dir)
        index_path = os.path.join(static_dir, "index.html")
        if not os.path.exists(index_path):
            with open(index_path, "w", encoding="utf-8") as f:
                f.write("""<!DOCTYPE html>
<html>
<head><title>Chat Saber Pro</title>
<style>body{font-family: sans-serif;}</style>
</head>
<body><h1>Chat Saber Pro RAG</h1><p>Tu interfaz de chat debería ir aquí.</p>
<p>Necesitarás JavaScript para conectar al WebSocket en /init.</p>
</body></html>""")
            logger.info(f"Archivo '{index_path}' de ejemplo creado. Reemplázalo con tu interfaz.")

    # Montar el directorio para que FastAPI sirva los archivos.
    app.mount(f"/{static_dir}", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Directorio estático '/{static_dir}' montado correctamente.")
except Exception as e:
     logger.error(f"No se pudo montar el directorio estático '{static_dir}'. Error: {e}", exc_info=True)
     exit()

# Ruta raíz: Redirige al archivo index.html dentro del directorio estático.
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root( request: Request ):
    logger.info("Acceso a la ruta raíz '/', redirigiendo a '/static/index.html'")
    return RedirectResponse(f"/{static_dir}/index.html")

# --- Funciones Auxiliares ---
async def safe_send_json(websocket: WebSocket, data: dict):
    """
    Función auxiliar para enviar mensajes JSON a través del WebSocket de forma segura.
    Verifica si la conexión WebSocket sigue activa antes de intentar enviar.
    Retorna True si el envío fue exitoso (o si la conexión ya estaba cerrada), False si ocurrió un error durante el envío.
    """
    if websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.send_json(data)
            logger.debug(f"Mensaje WebSocket enviado: {data.get('action', 'N/A')}")
            return True # Indica que el envío se intentó y no falló inmediatamente
        except (WebSocketDisconnect, ConnectionClosedOK, RuntimeError) as e:
            # Captura errores específicos de desconexión o cierre de conexión.
            logger.warning(f"No se pudo enviar mensaje WebSocket (cliente ya desconectado o error): {type(e).__name__} - {e}")
            return False # Indica que el envío falló porque la conexión estaba cerrada/rota.
    else:
        # Si el estado no es CONNECTED, no intentar enviar.
        logger.warning(f"Intento de enviar mensaje WebSocket ignorado, estado actual: {websocket.client_state}")
        return False # Indica que no se envió porque la conexión no estaba activa.

# --- Endpoint Principal del WebSocket para el Chat ---
@app.websocket("/init")
async def init_websocket_endpoint( websocket: WebSocket ):
    """
    Maneja las conexiones WebSocket entrantes para la funcionalidad del chat RAG.
    Mantiene un bucle para recibir mensajes del cliente y enviar respuestas generadas.
    """
    await websocket.accept()
    logger.info(f"Nueva conexión WebSocket aceptada desde: {websocket.client.host}:{websocket.client.port}")

    try:
        while True:
            # Espera a recibir un mensaje (historial de chat) del cliente en formato JSON.
            data = await websocket.receive_json()
            logger.debug(f"Mensaje JSON recibido: {json.dumps(data, indent=2)}")
            # Se puede agregar validación del JSON aquí
            # Envía una señal al frontend indicando que el backend está procesando la respuesta.
            await safe_send_json( websocket, { "action": "init_system_response", "status": "Processing..." } )
            # Llama a la función principal que implementa la lógica RAG.
            # Pasa el historial de mensajes, el objeto websocket y el prompt del sistema.
            await process_messages( data, websocket, system_prompt )
            # Envía una señal al frontend indicando que el backend ha terminado de enviar la respuesta.
            await safe_send_json( websocket, { "action": "finish_system_response", "status": "Done" } )

    except WebSocketDisconnect as e:
        # Maneja la desconexión iniciada por el cliente.
        logger.info(f"Cliente desconectado (WebSocketDisconnect - code: {e.code}) desde {websocket.client.host}:{websocket.client.port}.")
    except ConnectionClosedOK as e:
         # Maneja el cierre normal de la conexión.
         logger.info(f"Conexión WebSocket cerrada limpiamente (ConnectionClosedOK - code: {e.code}) desde {websocket.client.host}:{websocket.client.port}.")
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON recibido por WebSocket: {e}. Mensaje: {data}") # Loguear el dato problemático
        # Intentar enviar un mensaje de error al cliente si la conexión sigue activa.
        await safe_send_json(websocket, {"action": "error", "message": "Error en el formato del mensaje recibido."})
    except Exception as e:
        # Captura cualquier otra excepción durante el ciclo while.
        logger.error(f"Error inesperado en el bucle principal del WebSocket: {type(e).__name__} - {e}", exc_info=True) 
        # Intentar notificar al cliente si es posible.
        await safe_send_json(websocket, {"action": "error", "message": f"Ocurrió un error inesperado en el servidor: {type(e).__name__}"})

    finally:
        logger.info(f"Cerrando manejo para la conexión WebSocket desde {websocket.client.host}:{websocket.client.port}. Estado final: {websocket.client_state}")
        # FastAPI/Starlette manejan el cierre real de la conexión, no es necesario llamar a websocket.close() explícitamente aquí.


# --- Lógica Principal del RAG (Retrieval-Augmented Generation) ---
async def process_messages( messages: list, websocket: WebSocket, system_prompt: str ):
    """
    Función principal que orquesta el proceso RAG:
    1. Recupera contexto relevante de ChromaDB basado en el último mensaje.
    2. Augmenta el prompt para el LLM con el contexto recuperado y el historial.
    3. Genera la respuesta usando el LLM y la envía por streaming al cliente.
    """
    if not messages or not isinstance(messages, list):
        logger.warning("Se recibieron mensajes vacíos o en formato incorrecto. No se procesará.")
        await safe_send_json(websocket, {"action": "error", "message": "Mensaje inválido recibido."})
        return

    try:
        # Asume que el último mensaje en la lista es la consulta actual del usuario.
        last_user_message = messages[-1]["content"]
        if not isinstance(last_user_message, str) or not last_user_message.strip():
             logger.warning("El último mensaje del usuario está vacío o no es texto.")
             await safe_send_json(websocket, {"action": "info", "message": "Por favor, escribe tu pregunta."})
             return
        logger.info(f"Procesando consulta RAG para: '{last_user_message}'")
    except (IndexError, KeyError, TypeError) as e:
        logger.error(f"Error al acceder al último mensaje del usuario: {e}. Formato esperado: [{'role': 'user', 'content': '...'}]. Mensajes recibidos: {messages}")
        await safe_send_json(websocket, {"action": "error", "message": "Error interno al procesar el historial de mensajes."})
        return

    # --- 1. Recuperación (Retrieve) ---
    contexto_recuperado = "No se encontró información relevante en la base de datos para esta consulta." # Valor por defecto.
    retrieved_docs = [] # Inicializar lista de documentos recuperados.
    # retrieved_ids = [] 
    retrieved_distances = [] # Inicializar lista de Distancias

    try:
        # Consulta a ChromaDB para encontrar los fragmentos más similares al mensaje del usuario.
        results = collection.query(
            query_texts=[f"query: {last_user_message}"],
            n_results=2,
            include=["documents", "distances"]
        )
        # logger.debug(f"Resultados completos de ChromaDB: {results}")

        # Extrae los datos de los resultados y distancias.
        retrieved_docs = results.get("documents", [[]])[0]
        retrieved_distances = results.get("distances", [[]])[0]

        # *** Logging específico de los documentos recuperados (nivel DEBUG - Ajustado) ***
        if retrieved_docs:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("--- Documentos Recuperados de ChromaDB ---")
                context_list_str_for_llm = [] # Lista separada para construir el contexto del LLM
                # Iterar sobre distancias y documentos recuperados
                for i, (distance, doc) in enumerate(zip(retrieved_distances, retrieved_docs)):
                    cleaned_doc = doc.strip() if isinstance(doc, str) else ""
                    if cleaned_doc:
                        logger.debug(f"  [Doc {i+1}] Distancia: {distance:.4f}")
                        logger.debug(f"  Contenido: {cleaned_doc[:250]}...")
                        context_list_str_for_llm.append(f"--- Fragmento Contexto {i+1} ---\n{cleaned_doc}")
                    else:
                         logger.debug(f"  [Doc {i+1}] Distancia: {distance:.4f} - Contenido vacío o inválido.")

                logger.debug("-----------------------------------------")

                # Construir el string de contexto final para el LLM
                if context_list_str_for_llm:
                    contexto_recuperado = "\n\n".join(context_list_str_for_llm)
                    logger.info(f"Contexto recuperado ({len(context_list_str_for_llm)} fragmentos válidos) para el LLM.")
                    # logger.debug(f"Contexto Formateado Final para LLM:\n{contexto_recuperado}")
                else:
                    logger.warning("ChromaDB devolvió resultados, pero los fragmentos estaban vacíos/inválidos después de limpiar.")
                    contexto_recuperado = "Se encontraron referencias, pero el contenido asociado no pudo ser procesado."
            else:
                 # Si no estamos en DEBUG, solo formatear el contexto sin loguear detalles
                 context_list_str_for_llm = []
                 for doc in retrieved_docs:
                    cleaned_doc = doc.strip() if isinstance(doc, str) else ""
                    if cleaned_doc:
                        context_list_str_for_llm.append(f"---\n{cleaned_doc}")
                 if context_list_str_for_llm:
                    contexto_recuperado = "\n\n".join(context_list_str_for_llm)
                    logger.info(f"Contexto recuperado ({len(context_list_str_for_llm)} fragmentos válidos) para el LLM.")
                 else:
                    logger.warning("ChromaDB devolvió resultados, pero los fragmentos estaban vacíos/inválidos.")
                    contexto_recuperado = "Se encontraron referencias, pero el contenido asociado no pudo ser procesado."

        else:
             # Si ChromaDB no devuelve ningún documento en la lista 'documents'.
             logger.warning("ChromaDB no devolvió documentos relevantes (lista 'documents' vacía o ausente).")
             contexto_recuperado = "No se encontró información directamente relacionada en la base de datos para esta consulta."

    except Exception as e:
        # Captura errores durante la consulta a ChromaDB.
        logger.error(f"Error durante la consulta a ChromaDB: {type(e).__name__} - {e}", exc_info=True)
        contexto_recuperado = "Error técnico al intentar recuperar información de la base de datos."
        # Notifica al usuario sobre el problema de recuperación si la conexión sigue activa.
        await safe_send_json(websocket, {"action": "error", "message": "Hubo un problema técnico al buscar información relevante."})
        return

    # --- 2. Aumentación (Augment) ---
    # Combina el prompt del sistema (instrucciones + marcador de contexto) con el contexto recuperado.
    final_system_prompt = system_prompt + "\nCONTEXTO:\n" + contexto_recuperado
    # Crea la lista de mensajes final para enviar al LLM: incluye el system prompt y el historial completo de la conversación.
    prompt_messages = [ { "role": "system", "content": final_system_prompt } ] + messages
    # Loguear el prompt completo puede ser muy verboso, pero útil para depurar el comportamiento del LLM.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"--- Mensajes Completos para LLM ---")
        # Limitar la longitud del log para no saturar
        log_prompt = json.dumps(prompt_messages, indent=2, ensure_ascii=False)
        logger.debug(log_prompt[:2000] + "..." if len(log_prompt) > 2000 else log_prompt)
        logger.debug("----------------------------------")


    # --- 3. Generación (Generate) ---
    try:
        chat = model.start_chat(history=[])
        
        response = chat.send_message(
            final_system_prompt + "\n\n" + last_user_message,
            stream=True,
            generation_config=types.GenerationConfig(
                temperature=0.2,
                top_p=0.9,
            )
        )

        # Stream la respuesta al cliente
        for chunk in response:
            if chunk.text:
                if not await safe_send_json(websocket, { "action": "append_system_response", "content": chunk.text }):
                    logger.warning("Interrumpiendo streaming de respuesta LLM porque el cliente se desconectó.")
                    break

        logger.info("Respuesta del LLM enviada (o intento de envío finalizado si hubo desconexión).")

    except Exception as e:
        # Captura errores durante la llamada al LLM o durante el procesamiento del stream.
        logger.error(f"Error durante la llamada al LLM o el streaming de respuesta: {type(e).__name__} - {e}", exc_info=True)
        await safe_send_json(websocket, {"action": "error", "message": f"Lo siento, ocurrió un error técnico al generar la respuesta ({type(e).__name__})."})


# --- Punto de Entrada Principal para Ejecutar el Servidor ---
if __name__ == "__main__":
    logger.info("Iniciando servidor FastAPI con Uvicorn...")
    uvicorn.run(
        "main_rag:app",  # Indica a Uvicorn dónde encontrar la instancia de la app FastAPI.
        host="0.0.0.0",
        port=8000,
        reload=True,     # Habilita el modo de recarga automática para DESARROLLO.
        log_level="info" # Nivel de logging para UVICORN.
    )