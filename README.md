# CuruxIA - Predicción de Errores en Máquinas Industriales

## 📌 RESUMEN

CuruxIA se inspira en la excepcional audición de la curuxa (lechuza) para desarrollar una solución eficiente en la detección de fallos en maquinaria industrial. A través de sensores acústicos que envían datos a la nube de AWS, logramos identificar anomalías y notificar a los operarios para optimizar el funcionamiento de las máquinas.

## 🛠 INFRAESTRUCTURA

### 🔧 CONFIGURACIÓN DE HARDWARE Y CONECTIVIDAD

El sistema se basa en sensores acústicos conectados a microcontroladores ESP32, que transmiten datos por WiFi a la nube. Si se requiere preprocesamiento local, el sistema puede escalarse utilizando Raspberry Pi para **edge computing**.

![Infraestructura](attachment:5283ef65-8c53-45fa-a92e-a53f9682af5f:Captura_de_pantalla_1-4-2025_13416_docs.google.com.jpeg)

### ☁️ CLOUD AWS

AWS proporciona una infraestructura escalable y segura para almacenar y procesar datos. Los servicios clave incluyen:

- **Amazon SageMaker** – Ejecución del modelo de IA
- **Amazon RDS** – Almacenamiento de anomalías detectadas
- **Amazon ECS** – Alojamiento de la aplicación de visualización de errores y feedback de los operarios
- **Amazon QuickSight** – Analítica para tomar decisiones empresariales basadas en datos

![Cloud AWS](attachment:5283ef65-8c53-45fa-a92e-a53f9682af5f:Captura_de_pantalla_1-4-2025_13416_docs.google.com.jpeg)

## 🤖 METODOLOGÍA

El sistema procesa los datos acústicos en la nube aplicando **Descomposición en Valores Singulares (SVD)** para eliminar ruido ambiental y transforma las ondas sonoras en **espectrogramas MEL** para facilitar el entrenamiento del modelo de IA.

![Metodología](attachment:4381e6f2-2387-4a4e-9410-b10316beb88a:Captura_de_pantalla_1-4-2025_134124_docs.google.com.jpeg)

### 🔍 FASE DE ENTRENAMIENTO

El modelo de IA utiliza redes neuronales avanzadas:

1. **CNN** – Reconocimiento de patrones en imágenes
2. **LSTM** – Análisis de secuencias de sonido
3. **Autoencoders** – Detección de anomalías mediante compresión y reconstrucción de datos

Durante esta fase, los operarios proporcionarán feedback sobre las alertas de fallos, optimizando el modelo de manera continua.

### 🚀 FASE FINAL

Cuando el modelo alcance una alta precisión en sus predicciones, podrá operar de manera autónoma, permitiendo su implementación en zonas rurales con poca presencia de personal.

## 🔬 VIABILIDAD TÉCNICA

El proyecto presenta una alta viabilidad técnica debido a:

- **Hardware económico y fácil de instalar**
- **AWS como solución escalable y segura**
- **Modelo IA basado en investigaciones con problemáticas similares**

![Viabilidad](attachment:fc7743d7-bf06-4787-a3f0-7b1ce4933b3c:Captura_de_pantalla_1-4-2025_134139_docs.google.com.jpeg)

## 🌱 SOSTENIBILIDAD AMBIENTAL

CuruxIA utiliza sensores con componentes reemplazables para minimizar desperdicios y prolongar su vida útil mediante protección contra factores ambientales.

## 🏡 IMPACTO POSITIVO EN ZONAS RURALES

Facilita la detección de errores en infraestructuras críticas con poca presencia de personal, reduciendo costos operativos y asegurando el correcto funcionamiento de los sistemas de abastecimiento.

---

