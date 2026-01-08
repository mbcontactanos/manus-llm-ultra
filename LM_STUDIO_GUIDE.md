# 🚀 Guía de Ejecución en LM Studio

## Manus 1.6 ULTRA Lite en LM Studio

LM Studio es una aplicación de escritorio que permite ejecutar LLMs localmente. Esta guía te muestra cómo ejecutar Manus 1.6 ULTRA Lite.

---

## 📋 Requisitos

- **LM Studio** (descargar desde https://lmstudio.ai)
- **GPU**: NVIDIA con CUDA (recomendado) o CPU
- **RAM**: 16GB+ (8GB mínimo)
- **Almacenamiento**: 10GB disponibles
- **Python**: 3.8+ (para scripts)

---

## 🔧 Paso 1: Instalar LM Studio

1. Ve a https://lmstudio.ai
2. Descarga la versión para tu SO (Windows, macOS, Linux)
3. Instala la aplicación
4. Abre LM Studio

---

## 📥 Paso 2: Descargar Modelo

### Opción A: Desde HuggingFace (Recomendado)

1. En LM Studio, haz clic en **"Search models"**
2. Busca: `manus-llm-ultra-lite`
3. Selecciona la versión cuantizada (GGUF)
4. Haz clic en **"Download"**

### Opción B: Descargar Manualmente

```bash
# Descarga el modelo GGUF
wget https://huggingface.co/models/manus-llm-ultra-lite/resolve/main/model.gguf

# O usa git-lfs
git clone https://huggingface.co/manus-llm-ultra-lite
```

---

## ⚙️ Paso 3: Configurar Modelo en LM Studio

1. Una vez descargado, haz clic en **"Load model"**
2. Selecciona el modelo descargado
3. Configura parámetros:

```
GPU Layers: 30 (ajusta según tu GPU)
Context Length: 32768
Temperature: 0.7
Top-P: 0.9
Top-K: 50
```

4. Haz clic en **"Load"**

---

## 💬 Paso 4: Usar el Chat

### En la Interfaz de LM Studio

1. Ve a la pestaña **"Chat"**
2. Escribe tu pregunta
3. Presiona Enter o haz clic en **"Send"**

### Ejemplos de Preguntas

**Matemáticas:**
```
Resuelve esta ecuación: x² + 2x - 3 = 0
```

**Hardware:**
```
Explícame cómo funciona la caché de un procesador
```

**Software:**
```
¿Cuál es la diferencia entre un kernel y un driver?
```

**Código:**
```
Genera una función Python que calcule el factorial
```

**Diseño:**
```
Crea un diseño Figma para un dashboard
```

---

## 🔌 Paso 5: Usar API Local

LM Studio expone una API OpenAI-compatible en `http://localhost:8000`

### Python

```python
import requests

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "manus-llm-ultra-lite",
    "messages": [
        {"role": "user", "content": "¿Cuál es 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
}

response = requests.post(url, json=payload)
print(response.json()['choices'][0]['message']['content'])
```

### cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "manus-llm-ultra-lite",
    "messages": [{"role": "user", "content": "Hola"}],
    "temperature": 0.7
  }'
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'manus-llm-ultra-lite',
    messages: [{ role: 'user', content: '¿Hola?' }],
    temperature: 0.7
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

---

## 📊 Paso 6: Monitoreo de Rendimiento

### En LM Studio

1. Ve a **"Settings"** → **"Performance"**
2. Observa:
   - **GPU Memory**: Uso de VRAM
   - **CPU Usage**: Uso de CPU
   - **Tokens/sec**: Velocidad de inferencia
   - **Latency**: Tiempo de respuesta

### Métricas Esperadas

| Métrica | Valor |
|---|---|
| GPU Memory | 6-8GB |
| Tokens/sec | 50-100 |
| Latency | 10-50ms |
| CPU Usage | 20-40% |

---

## 🔧 Paso 7: Optimizaciones

### Para Mejor Rendimiento

1. **Aumentar GPU Layers**: Si tienes GPU potente
   ```
   GPU Layers: 40-50
   ```

2. **Reducir Context Length**: Si necesitas más velocidad
   ```
   Context Length: 8192 (en lugar de 32768)
   ```

3. **Usar Batch Processing**: Para múltiples consultas
   ```python
   # Procesa múltiples prompts
   prompts = ["Pregunta 1", "Pregunta 2", "Pregunta 3"]
   for prompt in prompts:
       # Envía a LM Studio
   ```

### Para Menos Consumo de Memoria

1. **Reducir GPU Layers**:
   ```
   GPU Layers: 10-20
   ```

2. **Usar CPU Mode**: Si no tienes GPU
   ```
   GPU Layers: 0
   ```

3. **Reducir Batch Size**:
   ```
   Batch Size: 1
   ```

---

## 🐛 Troubleshooting

### Problema: "Out of Memory"
**Solución:**
- Reduce GPU Layers
- Reduce Context Length
- Cierra otras aplicaciones

### Problema: "Modelo muy lento"
**Solución:**
- Aumenta GPU Layers
- Usa GPU en lugar de CPU
- Reduce Context Length

### Problema: "API no responde"
**Solución:**
- Verifica que LM Studio esté ejecutándose
- Comprueba que el puerto 8000 esté disponible
- Reinicia LM Studio

### Problema: "Modelo no carga"
**Solución:**
- Verifica que el archivo GGUF sea válido
- Comprueba espacio en disco
- Descarga nuevamente el modelo

---

## 📚 Recursos Adicionales

- [LM Studio Docs](https://lmstudio.ai/docs)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Manus GitHub](https://github.com/mbcontactanos/manus-llm-ultra)

---

## 🎯 Casos de Uso

### 1. Desarrollo Local
```bash
# Ejecuta LM Studio en background
lm-studio --headless --port 8000
```

### 2. Integración con Aplicaciones
```python
# Usa en tu app Python
import requests

def ask_manus(question):
    response = requests.post(
        'http://localhost:8000/v1/chat/completions',
        json={
            'model': 'manus-llm-ultra-lite',
            'messages': [{'role': 'user', 'content': question}]
        }
    )
    return response.json()['choices'][0]['message']['content']
```

### 3. Testing y Validación
```python
# Prueba el modelo
test_questions = [
    "¿Cuál es 2+2?",
    "Resuelve x² = 4",
    "¿Qué es Python?"
]

for q in test_questions:
    print(ask_manus(q))
```

---

## ✅ Verificación

Para verificar que todo funciona:

1. Abre LM Studio
2. Carga el modelo
3. Escribe: "¿Cuál es tu nombre?"
4. Deberías recibir: "Soy Manus 1.6 ULTRA Lite..."

---

**¡Tu LLM experto está listo en LM Studio! 🚀**

Para más ayuda, visita: https://github.com/mbcontactanos/manus-llm-ultra
