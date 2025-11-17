# BERT QA – Reglamento UPB

Entregable #3 de la materia **Analítica de Datos No Estructurados / Minería Multimedia**.

El objetivo es hacer el fine-tunning de un modelo **BERT en español** para responder preguntas sobre el reglamento estudiantil de la UPB a partir de un dataset .json.

## Contenido

- `question_answering.ipynb`: notebook principal de entrenamiento y pruebas.
- `models/`: pesos del modelo entrenado.

## Estructura del proyecto
```
.
├─ question_answering.ipynb
├─ qa-bert-upb-checkpoints/      # Checkpoints por época (Trainer)
├─ qa-bert-upb-model/            # Modelo y tokenizer final (save_pretrained)
├─ README.md
├─ requirements.txt             
└─ data/ (opcional si se decide guardar el JSON localmente)
```

## Instalación
```
git clone https://github.com/simon2326/bert-qa-upb.git
cd bert-qa-upb
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Inferencia fuera del notebook
Ejemplo mínimo:
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_dir = "qa-bert-upb-model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

contexto = "..."  # Fragmento del reglamento
pregunta = "¿Cuál es la nota mínima para aprobar un curso en la UPB?"
print(qa({"question": pregunta, "context": contexto}))
```

## Notas sobre recuperación (IR)
El modelo solo extrae spans; requiere un módulo de recuperación (BM25, embeddings, etc.) para seleccionar el contexto antes de la inferencia.

## Limitaciones
- Si la respuesta no está en el chunk, puede devolver texto irrelevante (fallback al token CLS).
- max_length y doc_stride afectan cobertura de contexto largo.


## Recarga rápida de checkpoints
Para continuar entrenamiento:
```python
from transformers import Trainer, TrainingArguments
# Cargar último checkpoint en qa-bert-upb-checkpoints/
# training_args = TrainingArguments(resume_from_checkpoint="qa-bert-upb-checkpoints/<checkpoint>")
```

## Licencias
- Modelo base: dccuchile/bert-base-spanish-wwm-cased (ver Hugging Face).
- Dataset FAQ reglamento: uso académico.

## Autor
Notebook: Simón Correa Marín (proyecto académico).