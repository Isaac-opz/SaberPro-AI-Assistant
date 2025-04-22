import os
import time
import asyncio
from statistics import mean
import json
import pandas as pd
import logging

from chromadb import PersistentClient
from openai import AsyncOpenAI
import evaluate

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración ---
DB_DIR = os.path.abspath("chroma_db_saberpro")
COLLECTION_NAME = "preguntas_frecuentes_saberpro"
ENDPOINT = "http://127.0.0.1:39281/v1"
MODEL = "llama3.2:3b"
K_RETRIEVAL = 2

# --- System Prompt (copia exacta de main_rag.py) ---
SYSTEM_PROMPT = """
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

A continuación se presenta el contexto relevante recuperado de la base de datos para la consulta actual:
"""

# --- Inicialización de clientes ---
try:
    client = PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    logger.info(f"Conectado a ChromaDB en '{DB_DIR}', colección '{COLLECTION_NAME}'. Documentos: {collection.count()}")
    if collection.count() == 0:
         logger.error("¡La colección ChromaDB está vacía! Ejecuta db_init.py primero.")
         exit()
except Exception as e:
    logger.error(f"Error al conectar con ChromaDB: {e}", exc_info=True)
    exit()

try:
    llm_client = AsyncOpenAI(base_url=ENDPOINT, api_key="not-needed")
    logger.info(f"Cliente OpenAI configurado para endpoint: {ENDPOINT}")
except Exception as e:
    logger.error(f"Error al configurar el cliente OpenAI: {e}", exc_info=True)
    exit()

# --- Funciones de evaluación ---

async def eval_retriever(test_data, k=K_RETRIEVAL):
    """Evalúa el rendimiento del componente de recuperación (Retriever)."""
    precisions, recalls, rranks, hits = [], [], [], []
    times = []
    logger.info(f"Iniciando evaluación del Retriever con k={k}...")
    for i, item in enumerate(test_data):
        query = item["query"]
        query_for_retrieval = f"query: {query}"
        relevant = set(item.get("relevant_ids", []))
        if not relevant:
            logger.warning(f"Item {i} ('{query[:50]}...') no tiene 'relevant_ids' definidos. Saltando cálculo de métricas para este item.")
            continue

        start = time.perf_counter()
        try:
            res = collection.query(
                query_texts=[query_for_retrieval],
                n_results=k,
                include=["distances"]
            )
            retrieved_ids = res.get("ids", [[]])[0]
            retrieved_distances = res.get("distances", [[]])[0]

        except Exception as e:
            logger.error(f"Error en ChromaDB query para '{query[:50]}...': {e}", exc_info=True)
            precisions.append(0.0)
            recalls.append(0.0)
            rranks.append(0.0)
            hits.append(0)
            times.append(time.perf_counter() - start)
            continue

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # **CORRECCIÓN 3: Obtener IDs directamente del resultado**
        retrieved_ids = res.get("ids", [[]])[0] # Obtiene la lista de IDs para la primera (y única) query
        # retrieved_distances = res.get("distances", [[]])[0]

        #--logging--
        logger.info(f"--- DEBUG Item {i} ---")
        logger.info(f"Query: '{query}'")
        logger.info(f"Expected Relevant IDs: {relevant}")
        logger.info(f"Retrieved IDs by ChromaDB (Top {k}): {retrieved_ids}")
        # logger.info(f"Retrieved Distances: {res.get('distances', [[]])[0]}")
        logger.info(f"----------------------")

        if not retrieved_ids:
             logger.warning(f"ChromaDB no retornó IDs para la query: '{query[:50]}...'")
             # Asignar 0 si no se recuperó nada
             precisions.append(0.0)
             recalls.append(0.0)
             rranks.append(0.0)
             hits.append(0)
             continue

        # Calcular métricas (sin cambios en la lógica, pero ahora `retrieved_ids` tiene valores)
        tp = len([doc_id for doc_id in retrieved_ids if doc_id in relevant])

        precision_k = tp / k
        recall_k = tp / len(relevant) # len(relevant) > 0 ya que chequeamos antes
        precisions.append(precision_k)
        recalls.append(recall_k)

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        rranks.append(rr)
        hits.append(1 if tp > 0 else 0)

        # Loguear resultados parciales
        # logger.debug(f"Query: '{query[:50]}...' | Retrieved: {retrieved_ids} | Relevant: {relevant} | P@{k}: {precision_k:.2f} | R@{k}: {recall_k:.2f} | RR: {rr:.2f}")

    logger.info("Evaluación del Retriever completada.")
    # Calcular promedios solo si hay resultados válidos
    num_valid_items = len(precisions)
    if num_valid_items == 0:
        logger.error("No se pudieron calcular métricas de retriever para ningún item.")
        return {f'precision@{k}': 0, f'recall@{k}': 0, f'MRR@{k}': 0, f'Hit@{k}': 0, 'avg_time_retrieve': 0}

    return {
        f'precision@{k}': mean(precisions),
        f'recall@{k}': mean(recalls),
        f'MRR@{k}': mean(rranks),
        f'Hit@{k}': mean(hits),
        'avg_time_retrieve': mean(times) if times else 0
    }


async def eval_generator(test_data, k_retrieval=K_RETRIEVAL):
    """Evalúa el rendimiento del componente de Generación (LLM) DENTRO DEL FLUJO RAG."""
    ems, bleus, rouges = [], [], []
    times_generate = [] 
    times_retrieve_for_gen = []

    logger.info(f"Iniciando evaluación del Generador (RAG) con k={k_retrieval} para contexto...")
    try:
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
    except Exception as e:
        logger.error(f"Error cargando métricas de 'evaluate': {e}. Asegúrate que 'evaluate', 'bleu', 'rouge_score', 'nltk', 'sacrebleu' estén instalados.")
        return {'ExactMatch': 0, 'BLEU': 0, 'ROUGE-L': 0, 'avg_time_generate': 0}

    for i, item in enumerate(test_data):
        query = item["query"]
        reference_answer = item.get("reference", "").strip()
        if not reference_answer:
             logger.warning(f"Item {i} ('{query[:50]}...') no tiene 'reference' answer. Saltando evaluación de generación.")
             continue

        # --- PASO 1: Recuperación (RAG - Retrieve) ---
        contexto_recuperado = "No se encontró información relevante."
        start_retrieve = time.perf_counter()
        try:
           
            query_for_retrieval = f"query: {query}"
            res = collection.query(
                query_texts=[query_for_retrieval],
                n_results=k_retrieval,
                include=["documents"]
            )
            retrieved_docs = res.get("documents", [[]])[0]

            # Formatear contexto (similar a main_rag.py, ajusta si es necesario)
            valid_docs = [doc.strip() for doc in retrieved_docs if doc and doc.strip()]
            if valid_docs:
                 contexto_recuperado = "\n\n".join([f"--- Fragmento Contexto {j+1} ---\n{doc}"
                                                 for j, doc in enumerate(valid_docs)])
                # logger.debug(f"Contexto para '{query[:50]}...':\n{contexto_recuperado[:300]}...")
            else:
                 logger.warning(f"No se recuperaron documentos válidos para '{query[:50]}...'")

        except Exception as e:
            logger.error(f"Error en ChromaDB query durante RAG para '{query[:50]}...': {e}", exc_info=True)
            contexto_recuperado = "Error técnico al recuperar contexto." # Informar al LLM del error
        elapsed_retrieve = time.perf_counter() - start_retrieve
        times_retrieve_for_gen.append(elapsed_retrieve)

        # --- PASO 2: Aumentación (RAG - Augment) ---
        # Combinar System Prompt, Contexto Recuperado y Query
        final_system_prompt_content = SYSTEM_PROMPT + "\nCONTEXTO:\n" + contexto_recuperado
        prompt_messages = [
            {"role": "system", "content": final_system_prompt_content},
            {"role": "user", "content": query}
        ]
        # logger.debug(f"Prompt final para LLM (query: '{query[:50]}...'): {json.dumps(prompt_messages, indent=2, ensure_ascii=False)[:500]}...")

        # --- PASO 3: Generación (RAG - Generate) ---
        start_generate = time.perf_counter()
        generated_answer = "" # Default en caso de error
        try:
            resp = await llm_client.chat.completions.create(
                model=MODEL,
                messages=prompt_messages,
                stream=False,
                temperature=0.0
                # max_tokens=
            )
            if resp.choices and resp.choices[0].message:
                generated_answer = resp.choices[0].message.content.strip()
            else:
                 logger.warning(f"Respuesta del LLM vacía o inesperada para '{query[:50]}...'")

        except Exception as e:
            logger.error(f"Error en LLM completion para '{query[:50]}...': {e}", exc_info=True)
            generated_answer = "[Error en la generación]"
        elapsed_generate = time.perf_counter() - start_generate
        times_generate.append(elapsed_generate)

        # logger.info(f"Query: '{query}'\nReference: '{reference_answer}'\nGenerated: '{generated_answer}'\n---") # Log para comparar manualmente

        # --- PASO 4: Evaluación de la Generación ---
        ems.append(int(generated_answer == reference_answer))

        # BLEU y ROUGE esperan listas de strings
        try:
            bleu_score = bleu_metric.compute(predictions=[generated_answer], references=[[reference_answer]])['bleu']
            bleus.append(bleu_score)
        except Exception as e:
            logger.error(f"Error calculando BLEU para '{query[:50]}...': {e}. Gen: '{generated_answer}', Ref: '{reference_answer}'")
            bleus.append(0.0)

        try:
            rouge_score = rouge_metric.compute(predictions=[generated_answer], references=[reference_answer])
            rouges.append(rouge_score['rougeL']) # Usar ROUGE-L
        except Exception as e:
             logger.error(f"Error calculando ROUGE para '{query[:50]}...': {e}. Gen: '{generated_answer}', Ref: '{reference_answer}'")
             rouges.append(0.0)

    logger.info("Evaluación del Generador (RAG) completada.")
    num_valid_items_gen = len(ems)
    if num_valid_items_gen == 0:
         logger.error("No se pudieron calcular métricas de generación para ningún item.")
         return {'ExactMatch': 0, 'BLEU': 0, 'ROUGE-L': 0, 'avg_time_generate': 0, 'avg_time_retrieve_for_gen': 0}

    return {
        'ExactMatch': mean(ems),
        'BLEU': mean(bleus),
        'ROUGE-L': mean(rouges),
        'avg_time_generate': mean(times_generate) if times_generate else 0,
        'avg_time_retrieve_for_gen': mean(times_retrieve_for_gen) if times_retrieve_for_gen else 0
    }


async def main():
    test_data_file = 'test_data.json'
    results_file = 'benchmark_results.csv'
    try:
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        logger.info(f"Cargados {len(test_data)} items de prueba desde '{test_data_file}'")
    except FileNotFoundError:
        logger.error(f"Error: El archivo de datos de prueba '{test_data_file}' no se encontró.")
        return
    except json.JSONDecodeError as e:
         logger.error(f"Error: El archivo '{test_data_file}' no es un JSON válido: {e}")
         return
    except Exception as e:
        logger.error(f"Error inesperado al cargar '{test_data_file}': {e}")
        return

    if not test_data:
         logger.error("El archivo de datos de prueba está vacío.")
         return

    # Ejecutar evaluación del retriever
    retr_metrics = await eval_retriever(test_data, k=K_RETRIEVAL)
    print("\n--- Retriever Metrics ---")
    print(json.dumps(retr_metrics, indent=4))

    # Ejecutar evaluación del generador (RAG)
    gen_metrics = await eval_generator(test_data, k_retrieval=K_RETRIEVAL)
    print("\n--- Generator Metrics (RAG) ---")
    print(json.dumps(gen_metrics, indent=4))

    # Combinar y guardar resultados
    all_metrics = {**retr_metrics, **gen_metrics}
    df = pd.DataFrame([all_metrics])

    # Reordenar columnas para legibilidad
    ordered_columns = [
        f'precision@{K_RETRIEVAL}', f'recall@{K_RETRIEVAL}', f'MRR@{K_RETRIEVAL}', f'Hit@{K_RETRIEVAL}',
        'avg_time_retrieve', 'avg_time_retrieve_for_gen', # Tiempos de retrieval
        'ExactMatch', 'BLEU', 'ROUGE-L', # Métricas de generación
        'avg_time_generate' # Tiempo de generación LLM
    ]
    final_columns = [col for col in ordered_columns if col in df.columns]
    df = df[final_columns]

    try:
        df.to_csv(results_file, index=False)
        print(f"\nResultados guardados en '{results_file}'")
    except Exception as e:
        logger.error(f"Error al guardar los resultados en '{results_file}': {e}")

if __name__ == '__main__':
    asyncio.run(main())