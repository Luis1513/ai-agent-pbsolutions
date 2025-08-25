# 🤖 PB RAG - Agente de IA con RAG para Punta Blanca Solutions

## 📋 Descripción del Proyecto

Este proyecto implementa un **agente de IA inteligente** que demuestra capacidades de **RAG (Retrieval-Augmented Generation)** utilizando tecnologías de Google Cloud Platform y LangGraph. El agente puede responder preguntas sobre Punta Blanca Solutions basándose en información extraída de su sitio web y LinkedIn.

### 🎯 Objetivo Principal
Crear un agente de IA que pueda:
1. **Recibir preguntas** a través de una API REST
2. **Buscar información relevante** en una base de conocimiento vectorial
3. **Generar respuestas** utilizando un LLM con el contexto recuperado
4. **Responder en formato JSON estructurado** con fuentes y nivel de confianza

## 🏗️ Arquitectura del Sistema

### Arquitectura General
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Node    │───▶│  Retrieval Node  │───▶│ Generation Node │───▶│  Output Node    │
│                 │    │                  │    │                 │    │                 │
│ • Validación    │    │ • Embeddings     │    │ • LLM Prompt    │    │ • Formato JSON  │
│ • Limpieza      │    │ • Pinecone Query │    │ • Context Build │    │ • Respuesta     │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
```

### Flujo de Datos
1. **Recepción**: API REST recibe pregunta del usuario
2. **Validación**: Se valida y limpia la pregunta
3. **Búsqueda**: Se genera embedding y se busca en Pinecone
4. **Generación**: Se construye contexto y se genera respuesta con LLM
5. **Formato**: Se estructura la respuesta en JSON con metadatos

## 🛠️ Stack Tecnológico

### Tecnologías Principales
- **FastAPI**: Framework web moderno y rápido para la API REST
- **LangGraph**: Orquestación del agente con flujo de trabajo definido
- **OpenAI GPT-4o-mini**: LLM para generación de respuestas
- **OpenAI text-embedding-3-small**: Modelo de embeddings optimizado
- **Pinecone**: Base de datos vectorial para búsqueda semántica
- **Python 3.11**: Versión estable y compatible con todas las dependencias

### Dependencias de Desarrollo
- **Black**: Formateador de código Python
- **Ruff**: Linter rápido para Python
- **Pytest**: Framework de testing

## 📁 Estructura del Proyecto

```
ai-agent/

├── app/                          # Aplicación principal
│   ├── api/                     # Endpoints de la API
│   │   └── main.py             # Aplicación FastAPI principal
│   ├── core/                    # Configuración central
│   │   └── settings.py         # Gestión de variables de entorno
│   └── graph/                   # Agente LangGraph
│       └── agent_graph.py      # Implementación del agente RAG

├── data/                        # Datos procesados
│   ├── *.json                  # Documentos originales de Punta Blanca
│   ├── processed_chunks.json   # Chunks de texto procesados
│   └── embeddings_processed.json # Embeddings generados

├── ingest/                      # Pipeline de procesamiento
│   ├── document_processor.py   # Procesamiento de documentos
│   ├── embedding_processor.py  # Generación de embeddings
│   ├── pinecone_uploader.py    # Carga a Pinecone
│   └── run_pipeline.py         # Orquestador del pipeline

├── scripts/                     # Scripts de utilidad
│   └── smoke_check.py          # Verificación de configuración

├── Dockerfile                   # Containerización
├── .dockerignore               # Archivos a excluir del Docker
├── requirements.txt             # Dependencias de Python
└── README.md                    # Este archivo
```

## 🔧 Decisiones Técnicas y Justificaciones

### 1. **Elección de OpenAI como LLM**
**Decisión**: Uso de GPT-4o-mini y text-embedding-3-small
**Justificación**: 
- **Créditos existentes**: Ya disponía de créditos en OpenAI, representando costo cero
- **Calidad probada**: GPT-4o-mini ofrece excelente relación calidad-precio
- **Consistencia**: Mismo proveedor para embeddings y generación asegura compatibilidad
- **API estable**: OpenAI tiene una de las APIs más estables y documentadas

### 2. **Modelo de Embeddings Small**
**Decisión**: text-embedding-3-small en lugar de text-embedding-3-large
**Justificación**:
- **Optimización de costos**: 50% menos costoso que el modelo large
- **Dimensiones adecuadas**: 1536 dimensiones son suficientes para búsqueda semántica
- **Performance**: Mantiene excelente calidad para casos de uso de RAG
- **Velocidad**: Generación más rápida de embeddings

### 3. **Tamaño de Chunks Optimizado**
**Decisión**: Chunk size de 750 caracteres con overlap de 150
**Justificación**:
- **Contexto preservado**: 750 caracteres permiten mantener oraciones completas
- **Overlap estratégico**: 150 caracteres aseguran continuidad entre chunks
- **Balance memoria-calidad**: Optimiza uso de tokens del LLM
- **Evita cortes abruptos**: Previene pérdida de contexto importante

### 4. **Arquitectura de Carpetas data/ingest**
**Decisión**: Separación clara entre datos crudos y pipeline de procesamiento
**Justificación**:
- **Datos estructurados**: Los JSONs contienen información limpia y organizada
- **Pipeline reutilizable**: El proceso de ingest puede aplicarse a nuevos datos
- **Separación de responsabilidades**: Datos vs. lógica de procesamiento
- **Facilita mantenimiento**: Estructura clara para futuras expansiones

### 5. **Uso de Pinecone como Vector Store**
**Decisión**: Pinecone en lugar de alternativas gratuitas como Chroma o FAISS
**Justificación**:
- **Escalabilidad**: Maneja grandes volúmenes de vectores eficientemente
- **Reranking avanzado**: Implementa BGE-reranker-v2-m3 para mejor relevancia
- **API robusta**: Interfaz estable y bien documentada
- **Integración nativa**: Funciona perfectamente con LangChain/LangGraph

### 6. **Implementación de Reranking**
**Decisión**: Uso de BGE-reranker-v2-m3 en Pinecone
**Justificación**:
- **Mejora significativa**: Reranking puede mejorar la relevancia en 20-30%
- **Costo-beneficio**: El costo adicional se compensa con mejor calidad de respuestas
- **Implementación nativa**: Pinecone lo maneja automáticamente
- **Sin complejidad adicional**: No requiere lógica adicional en el código

### 7. **Arquitectura LangGraph con 4 Nodos**
**Decisión**: Implementación secuencial simple en lugar de flujo complejo
**Justificación**:
- **Simplicidad**: Solución que funciona vs. complejidad innecesaria
- **Debugging fácil**: Cada nodo tiene responsabilidad única
- **Mantenibilidad**: Fácil de entender y modificar
- **Escalabilidad**: Fácil agregar nuevos nodos o modificar flujo

### 8. **FastAPI como Framework Web**
**Decisión**: FastAPI en lugar de Flask o Django
**Justificación**:
- **Performance**: Rendimiento superior para APIs
- **Type hints**: Validación automática con Pydantic
- **Documentación automática**: Swagger/OpenAPI generado automáticamente
- **Async support**: Preparado para operaciones asíncronas futuras

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.11+
- Docker (para containerización)
- Cuenta en OpenAI con API key
- Cuenta en Pinecone con API key

### Variables de Entorno
Crear archivo `.env` en la raíz del proyecto:
```bash
# OpenAI Configuration
OPENAI_API_KEY=tu_api_key_aqui
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Pinecone Configuration
PINECONE_API_KEY=tu_api_key_aqui
PINECONE_INDEX=agent-db
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Application Configuration
ENV=dev
```

### Instalación Local
```bash
# Clonar repositorio
git clone https://github.com/[tu-usuario]/ai-agent-pbsolutions.git
cd ai-agent-pbsolutions

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# Ejecutar pipeline de ingest
python -m ingest.run_pipeline

# Ejecutar aplicación
uvicorn app.api.main:app --reload
```

### Instalación con Docker
```bash
# Construir imagen
docker build -t pb-rag .

# Ejecutar contenedor
docker run -p 8000:8000 --env-file .env pb-rag
```

## 📊 Pipeline de Datos

### 1. **Procesamiento de Documentos**
```bash
python -m ingest.document_processor
```
- Lee archivos JSON de la carpeta `data/`
- Divide texto en chunks de 750 caracteres
- Mantiene metadatos (fuente, sección, ID)

### 2. **Generación de Embeddings**
```bash
python -m ingest.embedding_processor
```
- Genera embeddings para cada chunk
- Usa OpenAI text-embedding-3-small
- Almacena embeddings en formato JSON

### 3. **Carga a Pinecone**
```bash
python -m ingest.pinecone_uploader
```
- Carga embeddings a Pinecone en lotes
- Configura índice con dimensiones correctas
- Verifica carga exitosa

### 4. **Pipeline Completo**
```bash
python -m ingest.run_pipeline
```
- Ejecuta todo el proceso secuencialmente
- Proporciona feedback en tiempo real
- Maneja errores y continúa el proceso

## 🔍 Uso de la API

### Endpoint Principal: `/ask`
**POST** `/ask`

**Request Body:**
```json
{
  "question": "¿Qué servicios ofrece Punta Blanca?"
}
```

**Response:**
```json
{
  "answer": "Punta Blanca ofrece servicios de consultoría en IA, desarrollo de soluciones a medida...",
  "sources": [
    "https://www.puntablanca.ai/services",
    "https://www.puntablanca.ai/"
  ],
  "confidence": 0.85
}
```

### Health Check: `/healthz`
**GET** `/healthz`

**Response:**
```json
{
  "status": "ok",
  "env": "dev",
  "openai_model": "gpt-4o-mini",
  "pinecone_index": "agent-db"
}
```

### Ejemplos de Uso

#### Con curl
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "¿Quiénes son los fundadores de Punta Blanca?"}'
```

#### Con Python
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "¿Qué es AI Fast Track?"}
)

print(response.json())
```

## 🧪 Testing y Verificación

### Smoke Test
```bash
python scripts/smoke_check.py
```
Verifica:
- Conexión con OpenAI
- Conexión con Pinecone
- Creación/configuración de índice
- Operaciones básicas de vectores

### Tests Unitarios
```bash
pytest test/
```

### Verificación Manual
1. Ejecutar aplicación
2. Hacer pregunta de prueba
3. Verificar respuesta coherente
4. Verificar fuentes y confianza

## 🚀 Deployment en Cloud Run

### 1. **Construir y Taggear Imagen**
```bash
# Construir imagen
docker build -t pb-rag .

# Taggear para Google Container Registry
docker tag pb-rag gcr.io/[PROJECT-ID]/pb-rag

# Subir a GCR
docker push gcr.io/[PROJECT-ID]/pb-rag
```

### 2. **Deploy en Cloud Run**
```bash
gcloud run deploy pb-rag \
  --image gcr.io/[PROJECT-ID]/pb-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="OPENAI_API_KEY=[KEY],PINECONE_API_KEY=[KEY]"
```

### 3. **Verificar Deployment**
```bash
# Obtener URL del servicio
gcloud run services describe pb-rag --region us-central1

# Probar endpoint
curl -X POST "[URL]/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Test question"}'
```

## 📚 Recursos y Referencias

### Documentación Oficial
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Pinecone Documentation](https://docs.pinecone.io/)

### Tutoriales y Ejemplos
- [LangGraph Agent Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Google Cloud Run Quickstart](https://cloud.google.com/run/docs/quickstarts)

### Herramientas de Desarrollo
- [Postman](https://www.postman.com/) - Testing de API
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) - Containerización
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) - Deployment

## 🤝 Contribución

### Estructura de Commits
- `feat:` Nueva funcionalidad
- `fix:` Corrección de bugs
- `docs:` Documentación
- `refactor:` Refactorización de código
- `test:` Tests y testing

### Pull Request Process
1. Fork del repositorio
2. Crear rama feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit cambios: `git commit -m 'feat: agregar nueva funcionalidad'`
4. Push a rama: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

## 📄 Licencia

Este proyecto es parte de una prueba técnica para el puesto de AI Engineer en Punta Blanca Solutions.

## 👨‍💻 Autor

Desarrollado como prueba técnica para demostrar capacidades en:
- Implementación de RAG con LangGraph
- Integración de APIs de IA (OpenAI, Pinecone)
- Desarrollo de APIs REST con FastAPI
- Containerización y deployment en Google Cloud Platform
- Arquitectura de sistemas de IA escalables

---

**Nota**: Este proyecto está optimizado para funcionar dentro de los límites gratuitos de GCP y utiliza eficientemente los créditos disponibles en OpenAI para minimizar costos operativos.
