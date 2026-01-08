#!/usr/bin/env python3
"""
Descargador de Plantillas de Automatización desde GitHub
Manus 1.6 ULTRA MEGA - Contexto de Automatización
"""

import os
import json
import subprocess
from typing import Dict, List, Any
from datetime import datetime

class GitHubAutomationDownloader:
    """Descarga plantillas de automatización de GitHub"""
    
    def __init__(self):
        self.repos_to_download = [
            # n8n
            "n8n-io/n8n",
            "n8n-io/n8n-nodes-base",
            
            # Make (Integromat)
            "make-community/make-sdk",
            
            # Automatización
            "getgrav/grav",
            "huginn/huginn",
            
            # Flujos de trabajo
            "apache/airflow",
            "prefecthq/prefect",
            
            # Orquestación
            "kubernetes/kubernetes",
            "docker/docker",
            
            # Scripting
            "ansible/ansible",
            "saltstack/salt",
        ]
        
        self.downloaded_repos = []
        self.automation_templates = {}
    
    def download_repository(self, repo_url: str, max_depth: int = 2) -> bool:
        """Descarga repositorio de GitHub"""
        
        print(f"\n[GitHub] 📥 Descargando {repo_url}...")
        
        try:
            repo_name = repo_url.split('/')[-1]
            local_path = f"./automation_templates/{repo_name}"
            
            # Crear directorio
            os.makedirs(local_path, exist_ok=True)
            
            # Clonar repositorio (shallow clone para ahorrar espacio)
            cmd = [
                "git", "clone",
                "--depth", "1",
                f"https://github.com/{repo_url}.git",
                local_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0:
                print(f"[GitHub] ✅ Repositorio descargado: {repo_name}")
                self.downloaded_repos.append({
                    "repo": repo_url,
                    "local_path": local_path,
                    "timestamp": datetime.now().isoformat()
                })
                return True
            else:
                print(f"[GitHub] ⚠️  Error descargando: {result.stderr.decode()[:100]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[GitHub] ⏱️  Timeout descargando {repo_url}")
            return False
        except Exception as e:
            print(f"[GitHub] ❌ Error: {str(e)[:100]}")
            return False
    
    def extract_automation_patterns(self, repo_path: str) -> Dict[str, Any]:
        """Extrae patrones de automatización del repositorio"""
        
        patterns = {
            "workflows": [],
            "templates": [],
            "examples": [],
            "documentation": []
        }
        
        try:
            # Buscar archivos de flujo de trabajo
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Workflows
                    if file.endswith(('.yml', '.yaml')) and 'workflow' in root.lower():
                        patterns["workflows"].append(file_path)
                    
                    # Templates
                    elif 'template' in file.lower():
                        patterns["templates"].append(file_path)
                    
                    # Examples
                    elif file.startswith('example') or 'example' in file.lower():
                        patterns["examples"].append(file_path)
                    
                    # Documentation
                    elif file in ['README.md', 'GUIDE.md', 'TUTORIAL.md']:
                        patterns["documentation"].append(file_path)
                
                # Limitar búsqueda a 3 niveles
                if root.count(os.sep) - repo_path.count(os.sep) > 3:
                    dirs[:] = []
            
        except Exception as e:
            print(f"[Extract] ⚠️  Error extrayendo patrones: {str(e)[:50]}")
        
        return patterns
    
    def create_automation_knowledge_base(self) -> Dict[str, Any]:
        """Crea base de conocimiento de automatización"""
        
        print("\n[Knowledge Base] 📚 Creando base de conocimiento...")
        
        knowledge_base = {
            "automation_types": {
                "workflow_orchestration": {
                    "tools": ["n8n", "Make", "Zapier", "Airflow"],
                    "use_cases": ["ETL", "Data pipelines", "Event-driven workflows"],
                    "patterns": ["Sequential", "Parallel", "Conditional"]
                },
                "infrastructure_automation": {
                    "tools": ["Kubernetes", "Docker", "Terraform", "Ansible"],
                    "use_cases": ["Deployment", "Scaling", "Configuration management"],
                    "patterns": ["IaC", "Containerization", "Orchestration"]
                },
                "ci_cd": {
                    "tools": ["GitHub Actions", "GitLab CI", "Jenkins", "CircleCI"],
                    "use_cases": ["Testing", "Building", "Deployment"],
                    "patterns": ["Pipeline", "Trigger", "Stage"]
                },
                "monitoring_alerting": {
                    "tools": ["Prometheus", "Grafana", "ELK", "DataDog"],
                    "use_cases": ["Monitoring", "Alerting", "Logging"],
                    "patterns": ["Metrics", "Dashboards", "Alerts"]
                }
            },
            "integration_patterns": {
                "api_integration": {
                    "description": "Integración con APIs REST",
                    "example": "GET /api/users → Process → POST /api/results"
                },
                "webhook_integration": {
                    "description": "Integración basada en webhooks",
                    "example": "GitHub webhook → Process → Slack notification"
                },
                "database_integration": {
                    "description": "Integración con bases de datos",
                    "example": "Query DB → Transform → Load to warehouse"
                },
                "file_integration": {
                    "description": "Integración con archivos",
                    "example": "Read CSV → Process → Write JSON"
                }
            },
            "best_practices": [
                "Use version control for all workflows",
                "Implement error handling and retries",
                "Monitor workflow execution",
                "Document workflows clearly",
                "Use environment variables for secrets",
                "Test workflows in staging first",
                "Implement logging and debugging",
                "Use scheduled backups"
            ]
        }
        
        print("[Knowledge Base] ✅ Base de conocimiento creada")
        
        return knowledge_base
    
    def generate_automation_templates(self) -> Dict[str, str]:
        """Genera templates de automatización"""
        
        templates = {
            "n8n_workflow": """{
  "name": "Sample Workflow",
  "nodes": [
    {
      "name": "Start",
      "type": "n8n-nodes-base.start",
      "position": [250, 300]
    },
    {
      "name": "HTTP Request",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://api.example.com/data",
        "method": "GET"
      }
    },
    {
      "name": "Set Data",
      "type": "n8n-nodes-base.set",
      "parameters": {
        "values": {
          "string": [
            {
              "name": "processed",
              "value": "{{ $json }}"
            }
          ]
        }
      }
    }
  ],
  "connections": {
    "Start": {
      "main": [[{"node": "HTTP Request", "branch": 0, "index": 0}]]
    },
    "HTTP Request": {
      "main": [[{"node": "Set Data", "branch": 0, "index": 0}]]
    }
  }
}""",
            
            "github_actions_workflow": """name: Automation Workflow

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * *'

jobs:
  automate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run automation
        run: |
          python3 automation.py
      
      - name: Notify
        run: |
          curl -X POST ${{ secrets.WEBHOOK_URL }} \\
            -d '{"status": "completed"}'
""",
            
            "ansible_playbook": """---
- name: Automation Playbook
  hosts: all
  gather_facts: yes
  
  tasks:
    - name: Install dependencies
      apt:
        name: "{{ item }}"
        state: present
      loop:
        - python3
        - git
    
    - name: Clone repository
      git:
        repo: 'https://github.com/example/repo.git'
        dest: /opt/automation
    
    - name: Run automation
      shell: |
        cd /opt/automation
        python3 main.py
      register: result
    
    - name: Log results
      debug:
        msg: "{{ result.stdout }}"
""",
            
            "docker_compose": """version: '3.8'

services:
  automation:
    image: automation:latest
    environment:
      - API_KEY=${API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"
    restart: unless-stopped
    
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  redis:
    image: redis:6
    ports:
      - "6379:6379"

volumes:
  postgres_data:
"""
        }
        
        return templates
    
    def save_knowledge_base(self, knowledge_base: Dict[str, Any]):
        """Guarda base de conocimiento en JSON"""
        
        output_path = "./automation_knowledge_base.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
        
        print(f"[Save] ✅ Base de conocimiento guardada: {output_path}")

def demo():
    """Demostración"""
    
    print("\n" + "="*80)
    print("🤖 DESCARGADOR DE PLANTILLAS DE AUTOMATIZACIÓN")
    print("="*80)
    
    downloader = GitHubAutomationDownloader()
    
    # Descargar algunos repositorios
    print("\n[Step 1] Descargando repositorios...")
    for repo in downloader.repos_to_download[:3]:  # Demo con 3 repos
        downloader.download_repository(repo)
    
    # Crear base de conocimiento
    print("\n[Step 2] Creando base de conocimiento...")
    knowledge_base = downloader.create_automation_knowledge_base()
    
    # Generar templates
    print("\n[Step 3] Generando templates...")
    templates = downloader.generate_automation_templates()
    
    # Guardar
    print("\n[Step 4] Guardando...")
    downloader.save_knowledge_base(knowledge_base)
    
    # Mostrar información
    print("\n[Info] 📊 Repositorios descargados:")
    for repo in downloader.downloaded_repos:
        print(f"  ✓ {repo['repo']}")
    
    print("\n[Info] 📋 Templates disponibles:")
    for template_name in templates.keys():
        print(f"  ✓ {template_name}")
    
    print("\n" + "="*80)
    print("✅ Demostración completada")
    print("="*80)

if __name__ == "__main__":
    demo()
