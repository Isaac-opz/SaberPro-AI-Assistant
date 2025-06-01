## SaberPro AI Assistant

```
   _____       __              ____                ___    ____   ___              _      __              __ 
  / ___/____ _/ /_  ___  _____/ __ \_________     /   |  /  _/  /   |  __________(_)____/ /_____ _____  / /_
  \__ \/ __ `/ __ \/ _ \/ ___/ /_/ / ___/ __ \   / /| |  / /   / /| | / ___/ ___/ / ___/ __/ __ `/ __ \/ __/
 ___/ / /_/ / /_/ /  __/ /  / ____/ /  / /_/ /  / ___ |_/ /   / ___ |(__  |__  ) (__  ) /_/ /_/ / / / / /_  
/____/\__,_/_.___/\___/_/  /_/   /_/   \____/  /_/  |_/___/  /_/  |_/____/____/_/____/\__/\__,_/_/ /_/\__/  
                                                                                                            
```               

# 📚 SABERPRO-AI-ASSISTANT

Este es un asistente virtual inteligente basado en técnicas de Recuperación Aumentada por Generación (RAG), diseñado para responder preguntas relacionadas con las **Pruebas Saber Pro y TyT**. El asistente se comunica mediante una interfaz de chat amigable.

---

## 🚀 Características

- ✅ Asistente conversacional personalizado sobre Saber Pro y TyT.
- ✅ Interfaz web tipo chat moderna (HTML, CSS, JS).
- ✅ Backend en Python con integración de modelo LLM (Gemini API).
- ✅ Sistema RAG: búsqueda semántica con base de datos de contexto.

---

## 🛠 Requisitos

- **Python**: 3.12
- **Entorno virtual** (venv)
- Archivo `.env` con tu clave de API de Gemini.

---

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Isaac-opz/SaberPro-AI-Assistant.git
cd saberpro-ai-assistant
```

### 2. Crear y activar entorno virtual

```bash
python3 -m venv venv
# En Windows
venv\Scripts\activate
# En Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🔐 Configuración del entorno

Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```env
GEMINI_API_KEY=tu_clave_aquí
```

---

## ▶️ Ejecución

Lanza la aplicación con:

```bash
python .\main_rag.py
```

Esto ejecutará el backend y abrirá la interfaz web en tu navegador local. Desde ahí puedes interactuar con el asistente.

---


## 📘 Notas

- Este asistente está orientado exclusivamente a temas relacionados con las pruebas Saber Pro y TyT.
- La API de Gemini puede tener límites según el plan que uses. Asegúrate de usar una clave válida.

---


## 🤝 Agradecimientos

Desarrollado como solución de asistencia educativa para estudiantes que presentan las pruebas Saber Pro y TyT en Colombia.
