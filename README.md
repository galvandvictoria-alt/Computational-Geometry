# Computational Geometry

Repositorio de ejemplos y herramientas para geometría computacional aplicada a voxelización 3D y análisis de contornos.

## Estructura del repositorio

- `data/` - Carpeta para datos de entrada y salida, si se requiere preparar conjuntos de prueba.
- `docs/9_voxelization/` - Herramientas para convertir modelos 3D a voxeles usando `binvox`, analizar el tensor de inercia y exportar resultados.
  - `main.py` - Script principal de voxelización y análisis de inercia.
  - `tools.py` - Funciones auxiliares: conversión `binvox` → `numpy`, exportación a `OBJ`, ejecución de `binvox`.
  - `binvox` - Binario de voxelización usado por el pipeline.
- `docs/10_corners/` - Código para análisis de contornos en imágenes binarias y detección de puntos característicos mediante gramáticas libres de contexto.
  - `context-free_grammar.py` - Funciones de procesamiento de contornos, cálculo de AF8 y detección de break points.

## Qué hace este repositorio

- Convierte modelos 3D (por ejemplo, STL/OBJ) a una representación de voxeles.
- Calcula el centro de masa y el tensor de inercia de una nube de voxeles.
- Alinea el objeto con sus ejes principales de inercia.
- Exporta un resultado en formato `OBJ` para visualización.
- Incluye herramientas para extraer y codificar contornos de objetos binarios en imágenes.
- Implementa transformaciones AF8 y detección de puntos de ruptura basados en gramáticas de contorno.

## Requisitos

- Python 3.8+
- `numpy`
- `trimesh`
- `Pillow`
- `matplotlib`

## Instalación

1. Crear un entorno virtual (recomendado):

```bash
python -m venv venv
```

2. Activar el entorno virtual:

```bash
# Windows
venv\Scripts\activate
```

3. Instalar dependencias:

```bash
pip install numpy trimesh pillow matplotlib
```

## Uso

### Voxelización 3D

1. Coloca tu modelo 3D (`.stl`, `.obj`, etc.) en `docs/9_voxelization/`.
2. Ejecuta el script:

```bash
cd docs/9_voxelization
python main.py
```

3. El flujo realiza:
- conversión del modelo a `binvox`
- carga y conversión a `numpy`
- cálculo de centro de masa
- cálculo del tensor de inercia y sus valores propios
- alineación del objeto con los ejes principales
- exportación a un archivo `.obj` alineado

4. El resultado esperado es un archivo `*_alineado.obj` y la salida impresa en consola.

> Nota: el ejecutable `binvox` ya se encuentra en `docs/9_voxelization/binvox`.

### Análisis de contornos y gramática AF8

El código de `docs/10_corners/context-free_grammar.py` permite:

- cargar una imagen binaria
- extraer un contorno ordenado
- convertir ese contorno a código F8 y AF8
- detectar break points usando una gramática libre de contexto

Este módulo depende de `numpy`, `Pillow` y `matplotlib`.

## Notas

- El proyecto está en estado de ejemplo/prototipo.
- Si necesitas ejecutar una prueba completa en `docs/10_corners/`, puedes crear un script adicional para cargar una imagen y usar las funciones definidas en `context-free_grammar.py`.

## Licencia

Este repositorio incluye un archivo `LICENSE` en la raíz. Revisa ese archivo para conocer los términos de uso.
