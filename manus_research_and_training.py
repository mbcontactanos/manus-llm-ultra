#!/usr/bin/env python3
"""
Sistema de Investigación y Entrenamiento con Perplexity
Manus 1.6 ULTRA MEGA - YO (Manus) investigo, el LLM aprende
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

class ManusResearchEngine:
    """YO (Manus) investigo con Perplexity para entrenar el LLM"""
    
    def __init__(self):
        self.perplexity_api_key = os.getenv("SONAR_API_KEY", "")
        self.perplexity_endpoint = "https://api.perplexity.ai/chat/completions"
        self.research_topics = {
            "mathematics": [
                "Advanced calculus and differential equations",
                "Linear algebra and matrix operations",
                "Number theory and cryptography",
                "Complex analysis and topology"
            ],
            "hardware": [
                "CPU architecture and instruction sets",
                "Memory systems and cache hierarchy",
                "GPU computing and parallel processing",
                "FPGA and ASIC design"
            ],
            "software": [
                "Operating system design and kernels",
                "Compiler design and optimization",
                "Database systems and indexing",
                "Distributed systems and consensus"
            ],
            "engineering": [
                "Structural analysis and design",
                "Control systems and automation",
                "Signal processing and DSP",
                "Power systems and electrical grids"
            ],
            "automation": [
                "Workflow orchestration patterns",
                "ETL and data pipeline design",
                "Infrastructure as Code (IaC)",
                "CI/CD best practices"
            ]
        }
        
        self.research_results = []
        self.training_dataset = []
    
    def search_with_perplexity(self, query: str) -> Dict[str, Any]:
        """YO busco información con Perplexity"""
        
        print(f"\n[Manus Research] 🔍 Investigando: {query[:60]}...")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Proporciona información detallada y técnica sobre: {query}
                        
Incluye:
1. Definición y conceptos clave
2. Aplicaciones prácticas
3. Mejores prácticas
4. Ejemplos de código o implementación
5. Recursos para aprender más

Sé específico y técnico."""
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.perplexity_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                result = {
                    "query": query,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "source": "perplexity-api",
                    "citations": data.get('citations', [])
                }
                
                print(f"[Manus Research] ✅ Información recopilada ({len(content)} caracteres)")
                self.research_results.append(result)
                
                return result
            else:
                print(f"[Manus Research] ⚠️  Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[Manus Research] ❌ Error: {str(e)[:100]}")
            return None
    
    def conduct_research_campaign(self) -> List[Dict[str, Any]]:
        """YO conduzco campaña de investigación exhaustiva"""
        
        print("\n" + "="*80)
        print("🔬 CAMPAÑA DE INVESTIGACIÓN - MANUS")
        print("="*80)
        
        all_results = []
        
        for category, topics in self.research_topics.items():
            print(f"\n[Category] 📚 {category.upper()}")
            
            for topic in topics[:2]:  # Limitar a 2 por categoría para demo
                result = self.search_with_perplexity(topic)
                
                if result:
                    all_results.append(result)
        
        print(f"\n[Summary] ✅ Investigación completada: {len(all_results)} temas investigados")
        
        return all_results
    
    def create_training_dataset(self, research_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Creo dataset de entrenamiento a partir de investigación"""
        
        print("\n[Dataset Creation] 📊 Creando dataset de entrenamiento...")
        
        training_data = []
        
        for result in research_results:
            # Dividir contenido en chunks
            content = result['content']
            chunks = self._split_into_chunks(content, chunk_size=500)
            
            for i, chunk in enumerate(chunks):
                # Crear pares pregunta-respuesta
                qa_pair = {
                    "instruction": f"Explica sobre: {result['query']}",
                    "input": "",
                    "output": chunk,
                    "source": "perplexity-research",
                    "category": self._categorize_content(result['query']),
                    "timestamp": result['timestamp']
                }
                
                training_data.append(qa_pair)
        
        print(f"[Dataset Creation] ✅ Dataset creado: {len(training_data)} ejemplos")
        
        self.training_dataset = training_data
        return training_data
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Divide texto en chunks"""
        
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _categorize_content(self, query: str) -> str:
        """Categoriza contenido"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['math', 'calculus', 'algebra', 'equation']):
            return "mathematics"
        elif any(word in query_lower for word in ['cpu', 'gpu', 'memory', 'hardware', 'processor']):
            return "hardware"
        elif any(word in query_lower for word in ['software', 'kernel', 'os', 'driver', 'compiler']):
            return "software"
        elif any(word in query_lower for word in ['engineer', 'design', 'system', 'control']):
            return "engineering"
        elif any(word in query_lower for word in ['workflow', 'automation', 'pipeline', 'etl']):
            return "automation"
        else:
            return "general"
    
    def save_training_dataset(self, filename: str = "training_dataset.jsonl"):
        """Guardo dataset en formato JSONL para entrenamiento"""
        
        print(f"\n[Save] 💾 Guardando dataset: {filename}...")
        
        with open(filename, 'w', encoding='utf-8') as f:
            for item in self.training_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"[Save] ✅ Dataset guardado: {len(self.training_dataset)} ejemplos")
        
        return filename
    
    def generate_training_statistics(self) -> Dict[str, Any]:
        """Genero estadísticas del dataset"""
        
        stats = {
            "total_examples": len(self.training_dataset),
            "categories": {},
            "avg_output_length": 0,
            "total_tokens": 0,
            "sources": {}
        }
        
        total_length = 0
        
        for item in self.training_dataset:
            # Contar por categoría
            category = item.get('category', 'unknown')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Contar por fuente
            source = item.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
            
            # Calcular longitud promedio
            output_length = len(item.get('output', '').split())
            total_length += output_length
        
        if self.training_dataset:
            stats['avg_output_length'] = total_length / len(self.training_dataset)
            stats['total_tokens'] = total_length
        
        return stats
    
    def export_research_report(self, filename: str = "research_report.md"):
        """Exporto reporte de investigación"""
        
        print(f"\n[Report] 📄 Generando reporte: {filename}...")
        
        report = f"""# Reporte de Investigación Manus 1.6 ULTRA MEGA

## Resumen Ejecutivo

Fecha: {datetime.now().isoformat()}
Investigador: Manus (usando Perplexity API)
Temas Investigados: {len(self.research_results)}
Ejemplos de Entrenamiento: {len(self.training_dataset)}

## Temas Investigados

"""
        
        for i, result in enumerate(self.research_results, 1):
            report += f"\n### {i}. {result['query']}\n\n"
            report += f"**Fuente**: {result['source']}\n"
            report += f"**Timestamp**: {result['timestamp']}\n\n"
            report += f"**Contenido**:\n\n{result['content'][:500]}...\n\n"
        
        report += f"\n## Estadísticas del Dataset\n\n"
        
        stats = self.generate_training_statistics()
        report += f"- Total de ejemplos: {stats['total_examples']}\n"
        report += f"- Tokens totales: {stats['total_tokens']}\n"
        report += f"- Longitud promedio: {stats['avg_output_length']:.0f} palabras\n\n"
        
        report += f"### Por Categoría\n\n"
        for category, count in stats['categories'].items():
            report += f"- {category}: {count} ejemplos\n"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[Report] ✅ Reporte generado: {filename}")
        
        return filename

def demo():
    """Demostración"""
    
    print("\n" + "="*80)
    print("🔬 SISTEMA DE INVESTIGACIÓN CON PERPLEXITY")
    print("Manus investiga → Crea Dataset → Entrena LLM")
    print("="*80)
    
    engine = ManusResearchEngine()
    
    # Verificar API key
    if not engine.perplexity_api_key:
        print("\n⚠️  SONAR_API_KEY no configurada")
        print("Usando datos de demostración...\n")
        
        # Datos de demostración
        demo_results = [
            {
                "query": "Advanced calculus and differential equations",
                "content": """Calculus is the mathematical study of continuous change. It has two main branches:

1. Differential Calculus: Studies rates of change and slopes of curves
   - Derivatives measure instantaneous rate of change
   - Applications: optimization, physics, engineering
   
2. Integral Calculus: Studies accumulation and areas
   - Integrals reverse the process of differentiation
   - Applications: computing areas, volumes, work

Differential equations describe relationships between functions and their derivatives.
They're fundamental in physics, engineering, and biology.""",
                "timestamp": datetime.now().isoformat(),
                "source": "demo",
                "citations": []
            },
            {
                "query": "CPU architecture and instruction sets",
                "content": """CPU architecture defines how a processor executes instructions.

Modern CPUs use:
- Von Neumann Architecture: Single memory for code and data
- Instruction Set Architecture (ISA): x86-64, ARM, RISC-V
- Pipelining: Execute multiple instructions simultaneously
- Out-of-order execution: Reorder instructions for efficiency

Key components:
- ALU (Arithmetic Logic Unit): Performs calculations
- Registers: Fast temporary storage
- Cache: Multi-level memory hierarchy
- Control Unit: Coordinates operations""",
                "timestamp": datetime.now().isoformat(),
                "source": "demo",
                "citations": []
            }
        ]
        
        engine.research_results = demo_results
    else:
        # Investigación real con Perplexity
        engine.conduct_research_campaign()
    
    # Crear dataset
    print("\n[Step 2] Creando dataset de entrenamiento...")
    training_data = engine.create_training_dataset(engine.research_results)
    
    # Guardar dataset
    print("\n[Step 3] Guardando dataset...")
    engine.save_training_dataset()
    
    # Mostrar estadísticas
    print("\n[Step 4] Estadísticas del dataset...")
    stats = engine.generate_training_statistics()
    
    print(f"\n📊 Estadísticas:")
    print(f"  Total de ejemplos: {stats['total_examples']}")
    print(f"  Tokens totales: {stats['total_tokens']}")
    print(f"  Longitud promedio: {stats['avg_output_length']:.0f} palabras")
    print(f"\n  Por categoría:")
    for category, count in stats['categories'].items():
        print(f"    - {category}: {count}")
    
    # Exportar reporte
    print("\n[Step 5] Exportando reporte...")
    engine.export_research_report()
    
    print("\n" + "="*80)
    print("✅ Investigación completada")
    print("📁 Archivos generados:")
    print("  - training_dataset.jsonl (para entrenar el LLM)")
    print("  - research_report.md (reporte detallado)")
    print("="*80)

if __name__ == "__main__":
    demo()
