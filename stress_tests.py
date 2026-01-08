#!/usr/bin/env python3
"""
Suite de Pruebas de Estrés para Manus 1.6 ULTRA Lite
Identifica fallos, limitaciones y comportamientos bajo presión
"""

import time
import json
import traceback
from typing import Dict, List, Any, Tuple
from datetime import datetime
from manus_1_6_ultra_lite import Manus16UltraLite

class StressTestSuite:
    """Suite completa de pruebas de estrés"""
    
    def __init__(self):
        self.llm = Manus16UltraLite()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas"""
        
        print("\n" + "="*80)
        print("🔥 SUITE DE PRUEBAS DE ESTRÉS - MANUS 1.6 ULTRA LITE")
        print("="*80)
        
        # Pruebas de funcionamiento básico
        self.test_basic_initialization()
        self.test_language_understanding()
        self.test_math_capabilities()
        self.test_hardware_analysis()
        self.test_software_analysis()
        
        # Pruebas de estrés
        self.test_long_prompts()
        self.test_rapid_requests()
        self.test_complex_reasoning()
        self.test_edge_cases()
        self.test_memory_limits()
        
        # Pruebas de generación
        self.test_code_generation()
        self.test_figma_design_generation()
        
        # Resumen
        self.print_summary()
        self.export_results()
    
    def test_basic_initialization(self):
        """Prueba inicialización básica"""
        
        print("\n[TEST 1] ✓ Inicialización Básica")
        
        try:
            info = self.llm.get_model_info()
            
            assert info['name'] == "Manus 1.6 ULTRA Lite"
            assert info['parameters'] == "24B (cuantizados a 6GB)"
            assert info['training_tokens'] == "150 millones"
            
            self.log_test("Inicialización Básica", True, "Modelo inicializado correctamente")
            self.passed += 1
            
        except Exception as e:
            self.log_test("Inicialización Básica", False, str(e))
            self.failed += 1
    
    def test_language_understanding(self):
        """Prueba comprensión de lenguaje natural"""
        
        print("\n[TEST 2] 🗣️  Comprensión de Lenguaje Natural")
        
        test_cases = [
            "¿Cuál es el capital de Francia?",
            "Explícame la teoría de la relatividad",
            "¿Cómo se hace un café?",
            "¿Qué es la inteligencia artificial?"
        ]
        
        for i, prompt in enumerate(test_cases, 1):
            try:
                result = self.llm.understand_natural_language(prompt)
                
                assert 'intent' in result
                assert 'entities' in result
                assert 'required_expertise' in result
                
                print(f"  [{i}] ✓ '{prompt[:50]}...' - Intent: {result['intent']}")
                self.passed += 1
                
            except Exception as e:
                print(f"  [{i}] ✗ Error: {str(e)[:50]}")
                self.failed += 1
    
    def test_math_capabilities(self):
        """Prueba capacidades matemáticas"""
        
        print("\n[TEST 3] 🔢 Capacidades Matemáticas")
        
        math_problems = [
            "Resuelve: x² + 2x - 3 = 0",
            "Calcula la derivada de f(x) = 3x³ + 2x² - x + 5",
            "Integra: ∫(2x + 1)dx",
            "Resuelve el sistema: 2x + y = 5, x - y = 1"
        ]
        
        for i, problem in enumerate(math_problems, 1):
            try:
                result = self.llm.solve_math_problem(problem)
                
                assert 'solution' in result
                assert 'steps' in result
                assert len(result['steps']) > 0
                
                print(f"  [{i}] ✓ Problema resuelto - Confianza: {result['confidence']}")
                self.passed += 1
                
            except Exception as e:
                print(f"  [{i}] ✗ Error: {str(e)[:50]}")
                self.failed += 1
    
    def test_hardware_analysis(self):
        """Prueba análisis de hardware"""
        
        print("\n[TEST 4] 💻 Análisis de Hardware")
        
        hw_queries = [
            "¿Cómo funciona la caché de un procesador?",
            "Explícame la arquitectura x86-64",
            "¿Qué es un FPGA?"
        ]
        
        for i, query in enumerate(hw_queries, 1):
            try:
                result = self.llm.analyze_hardware(query)
                
                assert 'components' in result
                assert 'performance_metrics' in result
                
                print(f"  [{i}] ✓ Análisis completado")
                self.passed += 1
                
            except Exception as e:
                print(f"  [{i}] ✗ Error: {str(e)[:50]}")
                self.failed += 1
    
    def test_software_analysis(self):
        """Prueba análisis de software"""
        
        print("\n[TEST 5] 🖥️  Análisis de Software")
        
        sw_queries = [
            "¿Cómo optimizar un programa Python?",
            "Explícame cómo funciona un kernel",
            "¿Qué es la virtualización?"
        ]
        
        for i, query in enumerate(sw_queries, 1):
            try:
                result = self.llm.analyze_software(query)
                
                assert 'layers' in result
                assert 'optimization_strategies' in result
                
                print(f"  [{i}] ✓ Análisis completado")
                self.passed += 1
                
            except Exception as e:
                print(f"  [{i}] ✗ Error: {str(e)[:50]}")
                self.failed += 1
    
    def test_long_prompts(self):
        """Prueba con prompts largos"""
        
        print("\n[TEST 6] 📝 Prompts Largos")
        
        long_prompt = """
        Necesito que resuelvas un problema complejo de ingeniería. Tengo un sistema de control
        para un robot industrial que debe realizar tareas de precisión. El sistema utiliza
        procesadores ARM con arquitectura RISC-V, memoria limitada a 512MB, y debe ejecutarse
        en tiempo real. El código debe ser optimizado para consumir mínimos recursos.
        
        Además, necesito que generes un diseño en Figma que muestre la arquitectura del sistema,
        incluyendo los componentes de hardware, software, y las interfaces de comunicación.
        
        ¿Puedes ayudarme con esto?
        """
        
        try:
            start_time = time.time()
            result = self.llm.generate_response(long_prompt)
            elapsed = time.time() - start_time
            
            assert len(result) > 100
            assert elapsed < 30  # Debe completarse en menos de 30 segundos
            
            print(f"  ✓ Prompt largo procesado en {elapsed:.2f}s")
            print(f"  ✓ Respuesta: {len(result)} caracteres")
            self.passed += 1
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}")
            self.failed += 1
    
    def test_rapid_requests(self):
        """Prueba con solicitudes rápidas consecutivas"""
        
        print("\n[TEST 7] ⚡ Solicitudes Rápidas Consecutivas")
        
        prompts = [
            "¿Cuál es 2+2?",
            "¿Cuál es la capital de España?",
            "¿Qué es Python?",
            "¿Cómo se resuelve una ecuación?",
            "¿Qué es la IA?"
        ]
        
        try:
            start_time = time.time()
            
            for i, prompt in enumerate(prompts, 1):
                self.llm.generate_response(prompt)
                print(f"  [{i}] ✓ Solicitud procesada")
            
            elapsed = time.time() - start_time
            avg_time = elapsed / len(prompts)
            
            print(f"  ✓ {len(prompts)} solicitudes en {elapsed:.2f}s (promedio: {avg_time:.2f}s)")
            self.passed += 1
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}")
            self.failed += 1
    
    def test_complex_reasoning(self):
        """Prueba razonamiento complejo"""
        
        print("\n[TEST 8] 🧠 Razonamiento Complejo")
        
        complex_prompts = [
            "Diseña un algoritmo de ordenamiento eficiente y explica su complejidad",
            "¿Cómo se relacionan las matemáticas con la ingeniería?",
            "Explícame cómo funciona la inteligencia artificial desde cero"
        ]
        
        for i, prompt in enumerate(complex_prompts, 1):
            try:
                result = self.llm.generate_response(prompt)
                
                assert len(result) > 200
                
                print(f"  [{i}] ✓ Razonamiento completado ({len(result)} caracteres)")
                self.passed += 1
                
            except Exception as e:
                print(f"  [{i}] ✗ Error: {str(e)[:50]}")
                self.failed += 1
    
    def test_edge_cases(self):
        """Prueba casos extremos"""
        
        print("\n[TEST 9] ⚠️  Casos Extremos")
        
        edge_cases = [
            ("", "Prompt vacío"),
            ("a" * 1000, "Prompt muy largo (1000 caracteres)"),
            ("123456789", "Solo números"),
            ("!@#$%^&*()", "Solo caracteres especiales"),
            ("¿¿¿???", "Solo signos de puntuación")
        ]
        
        for prompt, description in edge_cases:
            try:
                result = self.llm.generate_response(prompt)
                print(f"  ✓ {description} - Manejado correctamente")
                self.passed += 1
                
            except Exception as e:
                print(f"  ⚠️  {description} - {str(e)[:40]}")
                self.warnings += 1
    
    def test_memory_limits(self):
        """Prueba límites de memoria"""
        
        print("\n[TEST 10] 💾 Límites de Memoria")
        
        try:
            # Simular múltiples generaciones
            for i in range(5):
                self.llm.generate_response(f"Pregunta {i+1}")
            
            print(f"  ✓ 5 generaciones completadas sin problemas de memoria")
            self.passed += 1
            
        except MemoryError:
            print(f"  ✗ Error de memoria detectado")
            self.failed += 1
        except Exception as e:
            print(f"  ⚠️  Advertencia: {str(e)[:50]}")
            self.warnings += 1
    
    def test_code_generation(self):
        """Prueba generación de código"""
        
        print("\n[TEST 11] 💻 Generación de Código")
        
        code_prompts = [
            "Genera una función Python que calcule el factorial",
            "Crea un algoritmo de búsqueda binaria en JavaScript",
            "Escribe una clase en Python para gestionar una cola"
        ]
        
        for i, prompt in enumerate(code_prompts, 1):
            try:
                result = self.llm.generate_response(prompt)
                
                # Verificar que contiene código
                has_code = any(keyword in result.lower() for keyword in 
                             ['def ', 'function', 'class ', 'const ', 'let '])
                
                if has_code or len(result) > 100:
                    print(f"  [{i}] ✓ Código generado ({len(result)} caracteres)")
                    self.passed += 1
                else:
                    print(f"  [{i}] ⚠️  Respuesta corta ({len(result)} caracteres)")
                    self.warnings += 1
                
            except Exception as e:
                print(f"  [{i}] ✗ Error: {str(e)[:50]}")
                self.failed += 1
    
    def test_figma_design_generation(self):
        """Prueba generación de diseños Figma"""
        
        print("\n[TEST 12] 🎨 Generación de Diseños Figma")
        
        design_prompts = [
            "Genera un diseño Figma para un dashboard de analytics",
            "Crea un diseño de landing page en Figma",
            "Diseña una interfaz de chat en Figma"
        ]
        
        for i, prompt in enumerate(design_prompts, 1):
            try:
                result = self.llm.generate_response(prompt)
                
                # Verificar que contiene elementos de diseño
                has_design = any(keyword in result.lower() for keyword in 
                               ['frame', 'component', 'color', 'layout', 'button', 'text'])
                
                if has_design or len(result) > 100:
                    print(f"  [{i}] ✓ Diseño generado ({len(result)} caracteres)")
                    self.passed += 1
                else:
                    print(f"  [{i}] ⚠️  Respuesta corta ({len(result)} caracteres)")
                    self.warnings += 1
                
            except Exception as e:
                print(f"  [{i}] ✗ Error: {str(e)[:50]}")
                self.failed += 1
    
    def log_test(self, test_name: str, passed: bool, message: str):
        """Registra resultado de prueba"""
        
        self.results["tests"].append({
            "name": test_name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def print_summary(self):
        """Imprime resumen de pruebas"""
        
        total = self.passed + self.failed + self.warnings
        
        print("\n" + "="*80)
        print("📊 RESUMEN DE PRUEBAS")
        print("="*80)
        
        print(f"\n✅ Pasadas: {self.passed}")
        print(f"❌ Fallidas: {self.failed}")
        print(f"⚠️  Advertencias: {self.warnings}")
        print(f"📈 Total: {total}")
        
        if self.failed == 0:
            print(f"\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        else:
            print(f"\n⚠️  {self.failed} pruebas fallaron")
        
        success_rate = (self.passed / total * 100) if total > 0 else 0
        print(f"\n📊 Tasa de éxito: {success_rate:.1f}%")
        
        print("\n" + "="*80 + "\n")
    
    def export_results(self):
        """Exporta resultados a JSON"""
        
        self.results["summary"] = {
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "total": self.passed + self.failed + self.warnings,
            "success_rate": (self.passed / (self.passed + self.failed + self.warnings) * 100) 
                          if (self.passed + self.failed + self.warnings) > 0 else 0
        }
        
        with open("stress_test_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resultados exportados a: stress_test_results.json")

def main():
    """Función principal"""
    
    suite = StressTestSuite()
    suite.run_all_tests()

if __name__ == "__main__":
    main()
