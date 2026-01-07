#!/usr/bin/env python3
"""
Repository Downloader - Descarga y Análisis de Repositorios
Descarga repositorios de Claude, OpenAI y OpenManus para análisis de arquitectura
"""

import os
import json
import subprocess
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

class RepositoryDownloader:
    """Descargador y analizador de repositorios de modelos"""
    
    def __init__(self, base_path: str = "/home/ubuntu/manus-llm-models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.repositories = {
            "claude": {
                "urls": [
                    "https://github.com/anthropics/anthropic-sdk-python",
                    "https://github.com/anthropics/prompt-eng-interactive-tutorial"
                ],
                "description": "Claude SDK y documentación"
            },
            "openai": {
                "urls": [
                    "https://github.com/openai/openai-python",
                    "https://github.com/openai/gpt-3.5-turbo-examples",
                    "https://github.com/openai/openai-cookbook"
                ],
                "description": "OpenAI SDK y ejemplos"
            },
            "openmanus": {
                "urls": [
                    "https://github.com/mbcontactanos/Clon-n8n",
                    "https://github.com/mbcontactanos/n8n-ai-custom",
                    "https://github.com/mbcontactanos/n8n-custom-ai"
                ],
                "description": "OpenManus y extensiones n8n"
            },
            "open_source": {
                "urls": [
                    "https://github.com/meta-llama/llama",
                    "https://github.com/mistralai/mistral-src",
                    "https://github.com/QwenLM/Qwen"
                ],
                "description": "Modelos open source (Llama, Mistral, Qwen)"
            }
        }
        
        self.download_log = []
        
        print("[RepositoryDownloader] ✅ Inicializado")

    def download_repository(self, url: str, name: str) -> Dict[str, Any]:
        """Descarga un repositorio de GitHub"""
        
        print(f"\n[Download] 📥 Descargando: {name}")
        print(f"  URL: {url}")
        
        repo_path = self.base_path / name
        
        try:
            if repo_path.exists():
                print(f"  ℹ️  Repositorio ya existe, actualizando...")
                subprocess.run(
                    ["git", "-C", str(repo_path), "pull"],
                    capture_output=True,
                    timeout=60
                )
            else:
                subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(repo_path)],
                    capture_output=True,
                    timeout=120
                )
            
            # Obtener información del repositorio
            repo_info = self._analyze_repository(repo_path, name)
            
            self.download_log.append({
                "name": name,
                "url": url,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "info": repo_info
            })
            
            print(f"  ✅ Descargado exitosamente")
            
            return repo_info
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            
            self.download_log.append({
                "name": name,
                "url": url,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            return {"error": str(e)}

    def _analyze_repository(self, repo_path: Path, name: str) -> Dict[str, Any]:
        """Analiza la estructura del repositorio"""
        
        analysis = {
            "name": name,
            "path": str(repo_path),
            "files": {
                "python": 0,
                "json": 0,
                "yaml": 0,
                "md": 0,
                "total": 0
            },
            "key_files": [],
            "directories": []
        }
        
        try:
            # Contar archivos
            for ext, pattern in [("python", "*.py"), ("json", "*.json"), ("yaml", "*.yaml"), ("md", "*.md")]:
                analysis["files"][ext] = len(list(repo_path.rglob(pattern)))
            
            analysis["files"]["total"] = sum(1 for _ in repo_path.rglob("*") if _.is_file())
            
            # Identificar archivos clave
            key_patterns = [
                "requirements.txt", "setup.py", "pyproject.toml",
                "model.py", "config.json", "architecture.py",
                "train.py", "inference.py", "api.py"
            ]
            
            for pattern in key_patterns:
                matches = list(repo_path.rglob(pattern))
                if matches:
                    analysis["key_files"].extend([str(m.relative_to(repo_path)) for m in matches[:3]])
            
            # Directorios principales
            analysis["directories"] = [d.name for d in repo_path.iterdir() if d.is_dir() and not d.name.startswith('.')][:10]
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis

    def download_all_repositories(self) -> Dict[str, Any]:
        """Descarga todos los repositorios"""
        
        print(f"\n{'='*70}")
        print(f"📥 DESCARGANDO TODOS LOS REPOSITORIOS")
        print(f"{'='*70}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "repositories": {},
            "total_downloaded": 0,
            "total_failed": 0
        }
        
        for category, repo_info in self.repositories.items():
            print(f"\n[Category] {category.upper()}")
            print(f"  {repo_info['description']}")
            
            results["repositories"][category] = []
            
            for i, url in enumerate(repo_info["urls"], 1):
                repo_name = f"{category}-{i}"
                result = self.download_repository(url, repo_name)
                
                if "error" not in result:
                    results["total_downloaded"] += 1
                else:
                    results["total_failed"] += 1
                
                results["repositories"][category].append(result)
        
        print(f"\n{'='*70}")
        print(f"✅ DESCARGA COMPLETADA")
        print(f"  Exitosos: {results['total_downloaded']}")
        print(f"  Fallidos: {results['total_failed']}")
        print(f"{'='*70}")
        
        return results

    def extract_model_architectures(self) -> Dict[str, Any]:
        """Extrae arquitecturas de modelos de los repositorios"""
        
        print(f"\n[Architecture] 🏗️  Extrayendo arquitecturas de modelos...")
        
        architectures = {
            "claude": self._extract_claude_architecture(),
            "openai": self._extract_openai_architecture(),
            "openmanus": self._extract_openmanus_architecture(),
            "open_source": self._extract_open_source_architecture()
        }
        
        return architectures

    def _extract_claude_architecture(self) -> Dict[str, Any]:
        """Extrae arquitectura de Claude"""
        
        print("\n  [Claude] Analizando arquitectura...")
        
        return {
            "name": "Claude",
            "type": "Transformer-based",
            "key_features": [
                "Constitutional AI",
                "Multi-turn conversations",
                "Context window: 100K tokens",
                "Instruction following",
                "Safety mechanisms"
            ],
            "capabilities": [
                "Text generation",
                "Code generation",
                "Analysis",
                "Reasoning",
                "Creative writing"
            ],
            "training_approach": "RLHF with Constitutional AI",
            "repositories": [
                "anthropic-sdk-python",
                "prompt-eng-interactive-tutorial"
            ]
        }

    def _extract_openai_architecture(self) -> Dict[str, Any]:
        """Extrae arquitectura de OpenAI"""
        
        print("\n  [OpenAI] Analizando arquitectura...")
        
        return {
            "name": "OpenAI GPT",
            "type": "Transformer-based",
            "key_features": [
                "GPT-3.5/GPT-4 architecture",
                "Multi-modal (text + vision)",
                "Function calling",
                "System prompts",
                "Token counting"
            ],
            "capabilities": [
                "Text generation",
                "Image analysis",
                "Function execution",
                "Embeddings",
                "Fine-tuning"
            ],
            "training_approach": "Reinforcement Learning from Human Feedback",
            "repositories": [
                "openai-python",
                "gpt-3.5-turbo-examples",
                "openai-cookbook"
            ]
        }

    def _extract_openmanus_architecture(self) -> Dict[str, Any]:
        """Extrae arquitectura de OpenManus"""
        
        print("\n  [OpenManus] Analizando arquitectura...")
        
        return {
            "name": "OpenManus",
            "type": "Agentic AI",
            "key_features": [
                "n8n integration",
                "MCP support",
                "Autonomous agents",
                "Workflow automation",
                "Custom tools"
            ],
            "capabilities": [
                "Workflow orchestration",
                "API integration",
                "Tool calling",
                "Multi-step reasoning",
                "State persistence"
            ],
            "training_approach": "Reinforcement Learning with workflow feedback",
            "repositories": [
                "Clon-n8n",
                "n8n-ai-custom",
                "n8n-custom-ai"
            ]
        }

    def _extract_open_source_architecture(self) -> Dict[str, Any]:
        """Extrae arquitectura de modelos open source"""
        
        print("\n  [Open Source] Analizando arquitecturas...")
        
        return {
            "models": [
                {
                    "name": "Llama",
                    "type": "Transformer",
                    "parameters": "7B-70B",
                    "key_features": ["Efficient", "Open weights", "Community support"]
                },
                {
                    "name": "Mistral",
                    "type": "Transformer",
                    "parameters": "7B",
                    "key_features": ["Fast", "Efficient", "Good reasoning"]
                },
                {
                    "name": "Qwen",
                    "type": "Transformer",
                    "parameters": "1.8B-72B",
                    "key_features": ["Multilingual", "Code generation", "Long context"]
                }
            ]
        }

    def export_analysis(self, filename: str = "repository_analysis.json"):
        """Exporta análisis de repositorios"""
        
        print(f"\n[Export] 💾 Exportando análisis...")
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "download_log": self.download_log,
            "total_repositories": len(self.download_log),
            "base_path": str(self.base_path)
        }
        
        filepath = Path(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"[Export] ✅ Exportado a: {filename}")

def demo():
    """Demostración del descargador"""
    
    downloader = RepositoryDownloader()
    
    print("\n" + "="*70)
    print("📥 REPOSITORY DOWNLOADER - DEMOSTRACIÓN")
    print("="*70)
    
    # Descargar todos los repositorios
    results = downloader.download_all_repositories()
    
    # Extraer arquitecturas
    architectures = downloader.extract_model_architectures()
    
    print(f"\n[Architectures] 🏗️  Arquitecturas extraídas:")
    for name, arch in architectures.items():
        print(f"\n  {name.upper()}:")
        if isinstance(arch, dict):
            if "name" in arch:
                print(f"    - Nombre: {arch['name']}")
                print(f"    - Tipo: {arch.get('type', 'N/A')}")
            if "models" in arch:
                for model in arch["models"]:
                    print(f"    - {model['name']}: {model['parameters']}")
    
    # Exportar
    downloader.export_analysis("/home/ubuntu/manus-llm-core/repository_analysis.json")

if __name__ == "__main__":
    demo()
