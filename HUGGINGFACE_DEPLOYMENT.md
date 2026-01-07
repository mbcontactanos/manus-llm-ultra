# 🚀 Guía de Despliegue en HuggingFace Spaces

## Manus 1.6 ULTRA Lite en HuggingFace Spaces (Gratuito)

### 📋 Requisitos Previos

- Cuenta en [HuggingFace](https://huggingface.co)
- Git instalado
- Token de HuggingFace

### 🔑 Paso 1: Obtener Token de HuggingFace

1. Ve a https://huggingface.co/settings/tokens
2. Crea un nuevo token con permisos de escritura
3. Copia el token

### 📁 Paso 2: Crear Space en HuggingFace

```bash
# Clona el repositorio
git clone https://github.com/mbcontactanos/manus-llm-ultra.git
cd manus-llm-ultra

# Configura credenciales de HuggingFace
huggingface-cli login
# Pega tu token cuando se solicite
```

### 🐳 Paso 3: Crear Dockerfile Optimizado

Crea `Dockerfile` en la raíz del proyecto:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos
COPY requirements.txt .
COPY manus_1_6_ultra_lite.py .
COPY app.py .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto
EXPOSE 7860

# Comando para ejecutar
CMD ["python", "app.py"]
```

### 🎯 Paso 4: Crear app.py para Gradio

Crea `app.py`:

```python
import gradio as gr
from manus_1_6_ultra_lite import Manus16UltraLite

# Inicializar modelo
llm = Manus16UltraLite()

def chat_with_manus(message: str, history: list) -> str:
    """Interfaz de chat con Manus"""
    
    # Procesar mensaje
    response = llm.generate_response(message)
    
    return response

def analyze_math(problem: str) -> str:
    """Analizar problema matemático"""
    
    result = llm.solve_math_problem(problem)
    return f"Solución: {result['answer']}\n\nExplicación: {result['explanation']}"

def analyze_hardware(query: str) -> str:
    """Analizar hardware"""
    
    result = llm.analyze_hardware(query)
    return f"Análisis: {result['performance_metrics']}"

def analyze_software(query: str) -> str:
    """Analizar software"""
    
    result = llm.analyze_software(query)
    return f"Estrategias: {result['optimization_strategies']}"

# Crear interfaz Gradio
with gr.Blocks(title="Manus 1.6 ULTRA Lite") as demo:
    gr.Markdown("# 🚀 Manus 1.6 ULTRA Lite - LLM Experto")
    gr.Markdown("Especialista en Matemáticas, Microinformática e Ingeniería")
    
    with gr.Tabs():
        # Tab 1: Chat General
        with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(label="Conversación")
            msg = gr.Textbox(
                label="Tu pregunta",
                placeholder="Escribe tu pregunta en lenguaje natural...",
                lines=2
            )
            submit_btn = gr.Button("Enviar")
            
            submit_btn.click(
                chat_with_manus,
                inputs=[msg, chatbot],
                outputs=chatbot
            )
        
        # Tab 2: Matemáticas
        with gr.TabItem("🔢 Matemáticas"):
            math_input = gr.Textbox(
                label="Problema matemático",
                placeholder="Ej: Resuelve la ecuación x² + 2x - 3 = 0",
                lines=3
            )
            math_output = gr.Textbox(
                label="Solución",
                interactive=False,
                lines=5
            )
            math_btn = gr.Button("Resolver")
            
            math_btn.click(
                analyze_math,
                inputs=math_input,
                outputs=math_output
            )
        
        # Tab 3: Hardware
        with gr.TabItem("💻 Hardware"):
            hw_input = gr.Textbox(
                label="Pregunta sobre hardware",
                placeholder="Ej: ¿Cómo funciona la caché de un procesador?",
                lines=3
            )
            hw_output = gr.Textbox(
                label="Análisis",
                interactive=False,
                lines=5
            )
            hw_btn = gr.Button("Analizar")
            
            hw_btn.click(
                analyze_hardware,
                inputs=hw_input,
                outputs=hw_output
            )
        
        # Tab 4: Software
        with gr.TabItem("🖥️ Software"):
            sw_input = gr.Textbox(
                label="Pregunta sobre software",
                placeholder="Ej: ¿Cómo optimizar un programa en Python?",
                lines=3
            )
            sw_output = gr.Textbox(
                label="Estrategias",
                interactive=False,
                lines=5
            )
            sw_btn = gr.Button("Analizar")
            
            sw_btn.click(
                analyze_software,
                inputs=sw_input,
                outputs=sw_output
            )
        
        # Tab 5: Información
        with gr.TabItem("ℹ️ Información"):
            info = llm.get_model_info()
            
            gr.Markdown(f"## {info['name']}")
            gr.Markdown(f"**Versión:** {info['version']}")
            gr.Markdown(f"**Parámetros:** {info['parameters']}")
            gr.Markdown(f"**Tokens de entrenamiento:** {info['training_tokens']}")
            gr.Markdown(f"**Contexto:** {info['context_window']}")
            
            gr.Markdown("### Especialidades")
            specialties_text = "\n".join([f"- {s.replace('_', ' ').title()}" for s in info['specialties']])
            gr.Markdown(specialties_text)
            
            gr.Markdown("### Requisitos de Recursos")
            resources = info['resource_requirements']
            resources_text = "\n".join([
                f"- **{k.replace('_', ' ').title()}:** {v}"
                for k, v in resources.items()
                if k != 'compatible_platforms'
            ])
            gr.Markdown(resources_text)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
```

### 📦 Paso 5: Actualizar requirements.txt

```
torch==2.1.0
transformers==4.35.0
gradio==4.0.0
bitsandbytes==0.41.0
peft==0.7.0
```

### 🚀 Paso 6: Crear Space en HuggingFace

```bash
# Crear repositorio de Space
git clone https://huggingface.co/spaces/TU_USUARIO/manus-lite
cd manus-lite

# Copiar archivos
cp ../manus-llm-ultra/* .

# Push a HuggingFace
git add .
git commit -m "Deploy Manus 1.6 ULTRA Lite"
git push
```

### ✅ Verificación

1. Ve a https://huggingface.co/spaces/TU_USUARIO/manus-lite
2. Espera a que se construya (5-10 minutos)
3. ¡Usa tu LLM!

### 📊 Monitoreo

- **Memoria GPU:** ~6GB
- **Memoria RAM:** ~4GB
- **Almacenamiento:** ~5GB
- **Tiempo de inicio:** ~30 segundos
- **Velocidad de inferencia:** 50-100 tokens/segundo

### 🔧 Optimizaciones Aplicadas

1. **Cuantización int4/nf4** - Reduce tamaño a 6GB
2. **LoRA** - Fine-tuning eficiente
3. **Flash Attention** - Inferencia rápida
4. **Arquitectura ligera** - 24B parámetros
5. **Caché inteligente** - Menos memoria

### 🐛 Troubleshooting

**Problema:** "Out of memory"
**Solución:** Reduce `max_new_tokens` en `app.py`

**Problema:** "Modelo muy lento"
**Solución:** Usa cuantización int4 en lugar de int8

**Problema:** "Gradio no inicia"
**Solución:** Verifica que el puerto 7860 esté disponible

### 📚 Recursos Adicionales

- [HuggingFace Spaces](https://huggingface.co/spaces)
- [Gradio Documentation](https://www.gradio.app/)
- [Transformers Library](https://huggingface.co/docs/transformers/)

---

**¡Tu LLM experto está listo en HuggingFace Spaces! 🎉**
