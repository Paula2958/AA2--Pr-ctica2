
# Práctica 2 — Autoencoders

**Asignatura:** Aprendizaje Automático II  
**Entorno:** Google Colab  
**Lenguaje:** Python  
**Framework:** PyTorch

---

## 1. Descripción del proyecto

Esta práctica examina la aplicación de los **autoencoders convolucionales** al procesamiento y reconstrucción de imágenes.

El conjunto de datos del **MNIST**, que consiste en imágenes de dígitos manuscritas, se utiliza para desarrollar dos tareas distintas.

1. **La eliminación de ruido (Denoising):** implica reconstruir la imagen a partir de una versión que ha sido degradada por el ruido gaussiano.
2. **La superresolución (Super-Resolution):** implica reconstruir imágenes de 28×28 píxeles desde sus contrapartes de menor resolución.

Ambas tareas nos permiten investigar cómo las redes neuronales adquieren representaciones útiles para recuperar información visual.

---

## 2. Objetivos

Los principales objetivos de la práctica son:

- Implementar autoencoders mediante PyTorch.
- Utilizar arquitecturas convolucionales para procesar imágenes.
- Entrenar modelos capaces de reconstruir imágenes degradadas.
- Evaluar el rendimiento calculando la media del error cuadrado (MSE).
- Comparar visualmente las imágenes originales y reconstruidas.
- Analizar la evolución del error durante el entrenamiento.

---

## 3. Tecnologías utilizadas

- **Python:** lenguaje de programación.
- **PyTorch:** implementación y entrenamiento de redes neuronales.
- **Torchvision:** descarga y preparación de MNIST.
- **NumPy:** operaciones numéricas.
- **Matplotlib:** visualización de imágenes y resultados.
- **Pandas:** almacenamiento de métricas en CSV.
- **Google Colab:** entorno de ejecución.
- **Google Drive:** almacenamiento persistente.

---

## 4. Dataset MNIST

Se utiliza el conjunto de datos MNIST, formado por imágenes en escala de grises de dígitos manuscritos del 0 al 9.

| Característica | Valor |
|---|---|
| Imágenes de entrenamiento | 60.000 |
| Imágenes de evaluación | 10.000 |
| Resolución original | 28×28 |
| Canales | 1 |
| Valores de píxeles | [0, 1] |
| Tamaño de lote | 64 |

Las imágenes se transforman en tensores mediante `transforms.ToTensor()`.

---

## 5. Parte I — Eliminación de ruido (Denoising)

### 5.1. Objetivo

El objetivo es entrenar a un autocodificador capaz de eliminar el ruido gaussiano de las imágenes MNIST.

Durante el entrenamiento, el modelo recibe una imagen degradada y aprende a reconstruir su versión.

### 5.2. Generación de ruido

Se añade ruido gaussiano a las imágenes mediante:

```math
\widetilde{x} = \operatorname{clip}(x + \varepsilon, 0, 1)
```

Donde:

- \(x\): imagen original.
- \(\widetilde{x}\): imagen con ruido.
- \(\varepsilon\): ruido gaussiano.
- \(\sigma=0.3\): desviación estándar del ruido.

Los valores se limitan al intervalo [0, 1].

### 5.3. Arquitectura

Se utiliza un autoencoder convolucional formado por un codificador y un decodificador.

**Encoder:**

```text
Input: 1 × 28 × 28
       ↓
Conv2d(1, 16, 3, stride=2)
       ↓
ReLU
       ↓
Conv2d(16, 32, 3, stride=2)
       ↓
ReLU
       ↓
Conv2d(32, 64, 7)
       ↓
Latent representation: 64 × 1 × 1
```

**Decoder:**

```text
Latent representation: 64 × 1 × 1
       ↓
ConvTranspose2d(64, 32, 7)
       ↓
ReLU
       ↓
ConvTranspose2d(32, 16, 3, stride=2)
       ↓
ReLU
       ↓
ConvTranspose2d(16, 1, 3, stride=2)
       ↓
Sigmoid
       ↓
Output: 1 × 28 × 28
```

El modelo aprende a transformar imágenes ruidosas en reconstrucciones próximas a las originales.

### 5.4. Configuración del entrenamiento

| Hiperparámetro | Valor |
|---|---|
| Épocas | 20 |
| Batch size | 64 |
| Optimizador | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Función de pérdida | MSE |
| Factor de ruido | 0.3 |
| Semilla | 42 |

Durante el entrenamiento:

1. Se obtienen imágenes originales.
2. Se añade ruido gaussiano.
3. Se introducen las imágenes ruidosas en el autoencoder.
4. Se comparan las reconstrucciones con las imágenes originales.
5. Se actualizan los pesos mediante retropropagación.

### 5.5. Resultados

Después de 20 épocas se obtuvieron los siguientes resultados:

| Métrica | Resultado |
|---|---:|
| MSE entrenamiento final | 0.00609 |
| MSE validación final | 0.00599 |
| MSE imágenes ruidosas | 0.046614 |
| MSE imágenes reconstruidas | 0.005993 |
| **Reducción del MSE** | **87.14 %** |

La reducción del error se calcula mediante:

\[
\text{Reducción}=
\frac{MSE_{\text{ruido}}-MSE_{\text{reconstrucción}}}
{MSE_{\text{ruido}}}\times100
\]

El autoencoder consigue una reducción del error cuadrático medio del **87.14 %**.

Las reconstrucciones muestran una eliminación significativa del ruido y conservan las características principales de los dígitos.

### 5.6. Archivos generados

```text
Practica2_Autoencoders/
├── data/
├── models/
│   └── denoising_autoencoder.pth
└── results/
    ├── original_vs_noisy.png
    ├── training_validation_loss.png
    ├── denoising_results.png
    ├── training_history.csv
    └── denoising_metrics.csv
```

---

## 6. Parte II — Superresolución

### 6.1. Objetivo

El objetivo de esta parte consiste en reconstruir imágenes MNIST de resolución original 28×28 a partir de imágenes de menor resolución.

Se consideran dos configuraciones:

- **7×7 → 28×28**
- **14×14 → 28×28**

Se entrenan modelos convolucionales para recuperar la información visual de los dígitos.

### 6.2. Preparación de imágenes

Las imágenes MNIST se redimensionan a sus resoluciones correspondientes.

Posteriormente, los modelos procesan imágenes de baja resolución para producir reconstrucciones de 28×28 píxeles.

Las imágenes sirven de referencia para calcular el error de reconstrucción.

### 6.3. Entrenamiento

Se utiliza el error cuadrático medio (MSE) para comparar las reconstrucciones con las imágenes originales.

En el notebook de superresolución se registraron las siguientes pérdidas al finalizar 10 épocas:

| Configuración | Pérdida registrada |
|---|---:|
| 7×7 → 28×28 | 0.0126 |
| 14×14 → 28×28 | 0.0015 |

**Nota:** estos valores corresponden a las pérdidas registradas en el último lote de entrenamiento. No deben interpretarse como métricas de validación sobre todo el conjunto de datos.

### 6.4. Análisis

Los resultados registrados indican que el modelo alcanza una tasa de error menor en comparación con el subconjunto de imágenes 14×14.

Esto se alinea con la observación de que las imágenes de mayor resolución conservan más detalles espaciales.

Para hacer una comparación cuantitativa definitiva entre los dos modelos, sería conveniente calcular el MEE medio sobre la misma base de evaluación.

---

## 7. Evaluación de los modelos

La evaluación se realiza mediante dos enfoques.

### 7.1. Evaluación cuantitativa

Se utiliza el error cuadrático medio:

\[
MSE=\frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{x}_i)^2
\]

Donde:

- \(x_i\): valor original del píxel.
- \(\hat{x}_i\): valor reconstruido.
- \(N\): número total de píxeles evaluados.

Un MSE menor indica una reconstrucción más próxima a la imagen original.

### 7.2. Evaluación visual

Se comparan tres tipos de imágenes:

1. Imágenes originales.
2. Imágenes degradadas.
3. Imágenes reconstruidas.

En el contexto de la eliminación del ruido, la degradación se modela añadiendo ruido gaussiano.

En la super-resolución, el objetivo es aumentar la resolución espacial de las imágenes.

---

## 8. Ejecución del proyecto

### Requisitos

Se recomienda ejecutar los notebooks en **Google Colab**, con un entorno que disponga de:

- Python 3
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pandas

Las dependencias pueden instalarse mediante:

```bash
pip install torch torchvision numpy matplotlib pandas
```

### Pasos de ejecución

1. Abrir el notebook correspondiente en Google Colab.
2. Si está disponible, active la GPU navegando hacia `Ejecución del entorno` y seleccionando `Cambio de tipo de entorno de ejecución '.
3. Ejecutar las celdas de importación y configuración.
4. Descargar y preparar MNIST.
5. Definir la arquitectura del modelo.
6. Configurar los hiperparámetros.
7. Ejecutar el entrenamiento.
8. Visualizar y evaluar las reconstrucciones.
9. Guardar los resultados obtenidos.

**Nota:** La función de eliminación de ruido almacena sus resultados en Google Drive. Se requiere autorización antes de ejecutar la primera celda.

---

## 9. Organización del trabajo

La práctica se divide en dos implementaciones complementarias:

| Parte | Responsabilidad |
|---|---|
| Eliminación de ruido | Implementación, entrenamiento y evaluación del Denoising Autoencoder |
| Superresolución | Implementación y entrenamiento de los modelos 7×7 y 14×14 |

Ambas partes utilizan MNIST y arquitecturas basadas en redes neuronales convolucionales.

---

## 10. Conclusiones

Este estudio investiga la aplicación de autoencoders a dos tareas de reconstrucción de imágenes: eliminación de ruido y superresolución.

En la tarea de **eliminación de ruido**, el modelo evolutivo redujo con éxito el MSE en **87. 14%**, logrando reconstrucciones visualmente muy cercanas a las imágenes.

En la tarea de **superresolución**, los resultados de entrenamiento registrados indican que el procesamiento de imágenes 14×14 muestra una tasa de error ligeramente menor en comparación con las imágenes 7×7.

Los experimentos demuestran que los autoencoders son capaces de aprender representaciones útiles para recuperar información visual de imágenes degradadas.

Como posibles mejoras futuras se plantean:

- Evaluar distintas arquitecturas convolucionales.
- Comparar diferentes niveles de ruido.
- Incorporar métricas adicionales como PSNR o SSIM.
- Evaluar la superresolución sobre un conjunto independiente.
- Experimentar con distintos tamaños de representación latente.

---

## 11. Estado del proyecto

- [x] Implementación del autoencoder de eliminación de ruido.
- [x] Entrenamiento durante 20 épocas.
- [x] Evaluación visual de las reconstrucciones.
- [x] Evaluación cuantitativa del denoising.
- [x] Guardado del modelo y métricas.
- [x] Implementación de modelos de superresolución.
- [x] Entrenamiento de las configuraciones 7×7 y 14×14.
- [ ] Revisión final de reproducibilidad de superresolución.
- [ ] Integración definitiva de ambas partes para la entrega.
