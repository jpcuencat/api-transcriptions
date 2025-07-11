#!/bin/bash

# Script para ejecutar diferentes tipos de pruebas

echo "=== Script de Pruebas de la API de Transcripción ==="

# Activar entorno virtual
source venv/bin/activate

# Función para mostrar ayuda
show_help() {
    echo "Uso: $0 [OPCIÓN]"
    echo ""
    echo "Opciones:"
    echo "  unit          Ejecutar solo pruebas unitarias (rápidas)"
    echo "  integration   Ejecutar solo pruebas de integración"
    echo "  api           Ejecutar solo pruebas de API"
    echo "  fast          Ejecutar pruebas rápidas para desarrollo"
    echo "  slow          Ejecutar pruebas lentas (incluye video real)"
    echo "  all           Ejecutar todas las pruebas"
    echo "  coverage      Ejecutar con reporte de cobertura"
    echo "  help          Mostrar esta ayuda"
}

# Función para ejecutar pruebas unitarias
run_unit_tests() {
    echo "🧪 Ejecutando pruebas unitarias..."
    pytest tests/test_unit.py -v -m "not slow"
}

# Función para ejecutar pruebas de integración
run_integration_tests() {
    echo "🔗 Ejecutando pruebas de integración..."
    pytest tests/test_integration.py -v -m "integration and not slow"
}

# Función para ejecutar pruebas de API
run_api_tests() {
    echo "🌐 Ejecutando pruebas de API..."
    pytest tests/ -v -m "api and not slow"
}

# Función para ejecutar pruebas rápidas
run_fast_tests() {
    echo "⚡ Ejecutando pruebas rápidas..."
    pytest tests/ -v -m "not slow and not integration"
}

# Función para ejecutar pruebas lentas
run_slow_tests() {
    echo "🐌 Ejecutando pruebas lentas (incluye video real)..."
    pytest tests/ -v -m "slow or requires_video"
}

# Función para ejecutar todas las pruebas
run_all_tests() {
    echo "🚀 Ejecutando todas las pruebas..."
    pytest tests/ -v
}

# Función para ejecutar con cobertura
run_coverage_tests() {
    echo "📊 Ejecutando pruebas con reporte de cobertura..."
    pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing
}

# Procesar argumentos
case "${1:-all}" in
    unit)
        run_unit_tests
        ;;
    integration)
        run_integration_tests
        ;;
    api)
        run_api_tests
        ;;
    fast)
        run_fast_tests
        ;;
    slow)
        run_slow_tests
        ;;
    all)
        run_all_tests
        ;;
    coverage)
        run_coverage_tests
        ;;
    help)
        show_help
        ;;
    *)
        echo "❌ Opción no válida: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "✅ Ejecución de pruebas completada."