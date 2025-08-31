from fastapi import FastAPI, Depends, HTTPException
from app.models import AnalyzeIn, AnalyzeOut, Sentiment, Urgency, ChatRequest, ChatResponse
from app.db import SessionLocal, Task, ChatMessage
from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

app = FastAPI()

# Inicializa el cliente con tu API Key (ya configurada en variable de entorno)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {"message": "API de análisis de solicitudes con LLM y FastAPI"}

@app.get("/tasks")
async def get_tasks(db=Depends(get_db)):
    tasks = db.query(Task).all()
    return [
        {
            "id": task.id,
            "solicitud": task.solicitud,
            "created_at": task.created_at,
            "suggested_response": task.suggested_response,
            "sentiment": task.sentiment,
            "urgency": task.urgency,
            "tags": task.tags.split(",")  # Convertimos de string a lista al devolver
        }
        for task in tasks
    ]

@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(data: AnalyzeIn, db=Depends(get_db)):
    # El prompt del sistema ahora instruye al modelo a devolver un JSON con campos específicos,
    # utilizando los valores de cadena exactos que esperan tus Enums.
    system_prompt_content = f"""
    Eres un analizador experto de incidencias. Tu tarea es analizar el texto de una solicitud o incidencia
    y proporcionar una respuesta sugerida, clasificar el sentimiento, la urgencia y generar tags relevantes.
    La salida debe ser un objeto JSON con las siguientes claves:
    - 'suggested_response': Una respuesta concisa y útil para la incidencia.
    - 'sentiment': Clasifica el sentimiento usando uno de estos valores: {', '.join([f"'{s.value}'" for s in Sentiment])}.
    - 'urgency': Clasifica la urgencia usando uno de estos valores: {', '.join([f"'{u.value}'" for u in Urgency])}.
    - 'tags': Una lista de cadenas con palabras clave relevantes para la incidencia.

    Ejemplo de formato JSON esperado:
    {{
        "suggested_response": "Hemos recibido su reporte sobre la interrupción del servicio. Estamos investigando la causa.",
        "sentiment": "{Sentiment.negative.value}",
        "urgency": "{Urgency.high.value}",
        "tags": ["servicio", "interrupción", "soporte", "red"]
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": data.solicitud}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    output_json_string = response.choices[0].message.content

    try:
        analysis_result = json.loads(output_json_string)

        # Convertimos las cadenas de la respuesta JSON a los tipos Enum
        # Usamos .get() con valores por defecto que también son Enums,
        # o que pueden ser manejados por el constructor del Enum.
        parsed_sentiment_str = analysis_result.get("sentiment", "neutral") 
        parsed_urgency_str = analysis_result.get("urgency", "low")

        # Intentamos convertir a Enum. Si falla, usamos un valor por defecto seguro (aunque sea 'neutral' o 'low').
        # Es mejor asegurar que los valores por defecto también sean parte del Enum si es posible.
        
        sentiment_enum = Sentiment(parsed_sentiment_str) if parsed_sentiment_str in [s.value for s in Sentiment] else Sentiment.neutral
        urgency_enum = Urgency(parsed_urgency_str) if parsed_urgency_str in [u.value for u in Urgency] else Urgency.low

        task = Task(
            solicitud=data.solicitud, # Guardamos la solicitud original
            suggested_response=analysis_result.get("suggested_response", "No se pudo generar una respuesta sugerida."), # Valor por defecto seguro
            sentiment=sentiment_enum.value, # Guardamos el valor del Enum
            urgency=urgency_enum.value, # Guardamos el valor del Enum
            tags=",".join(analysis_result.get("tags", ["general"])) # Guardamos tags como string separado por comas
        )

        db.add(task)
        db.commit()
        db.refresh(task)

        return AnalyzeOut(
            suggested_response=task.suggested_response, # Devolvemos la respuesta sugerida guardada
            sentiment=sentiment_enum, # Convertimos de nuevo a Enum al devolver
            urgency=urgency_enum, # Convertimos de nuevo a Enum al devolver
            tags=task.tags.split(",") # Convertimos de nuevo a lista al devolver
        )
    except (json.JSONDecodeError, ValueError) as e:
        # Manejar errores de JSON o si el valor del modelo no coincide con un miembro del Enum
        print(f"Error al procesar la respuesta del modelo: {e}")
        return AnalyzeOut(
            suggested_response="Error al procesar la respuesta del modelo.",
            sentiment=Sentiment.neutral, # Valor por defecto seguro
            urgency=Urgency.low,       # Valor por defecto seguro
            tags=["error_parsing"]
        )
    

@app.post("/incidencias/{id}/chat", response_model=ChatResponse)
async def chat_incidencia(id: int, chat: ChatRequest, db=Depends(get_db)):
    # Recupera la incidencia
    task = db.query(Task).filter(Task.id == id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Incidencia no encontrada")

    # Construye el contexto inicial con la incidencia
    system_prompt = f"""
    Eres un asistente experto en incidencias. El usuario consulta sobre la siguiente incidencia:
    ---
    Solicitud: {task.solicitud}
    Respuesta sugerida: {task.suggested_response}
    Sentimiento: {task.sentiment}
    Urgencia: {task.urgency}
    Tags: {task.tags}
    ---
    Responde de forma clara y útil a las dudas del usuario sobre esta incidencia.
    """

    # Prepara el historial de mensajes (máximo 5)
    history = chat.messages[-5:] if len(chat.messages) > 5 else chat.messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    for m in history:
        messages.append({"role": m.role, "content": m.content})
    # Añade el nuevo mensaje del usuario
    messages.append({"role": "user", "content": chat.user_message})

    # Guarda el mensaje del usuario en la base de datos
    user_msg = ChatMessage(
        incidencia_id=id,
        role="user",
        content=chat.user_message
    )
    db.add(user_msg)
    db.commit()

    # Llama al modelo
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )
    answer = response.choices[0].message.content

    # Guarda la respuesta del assistant en la base de datos
    assistant_msg = ChatMessage(
        incidencia_id=id,
        role="assistant",
        content=answer
    )
    db.add(assistant_msg)
    db.commit()

    return ChatResponse(response=answer)

@app.get("/incidencias/{id}/chat")
async def get_chat_incidencia(id: int, db=Depends(get_db)):
    # Recupera los mensajes de chat asociados a la incidencia, ordenados por timestamp
    messages = db.query(ChatMessage).filter(ChatMessage.incidencia_id == id).order_by(ChatMessage.timestamp).all()
    # Devuelve una lista de mensajes con rol y contenido
    return [
        {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()} for msg in messages
    ]