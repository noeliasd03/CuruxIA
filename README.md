# CuruxIA – Detección acústica de anomalías en maquinaria industrial

CuruxIA es un sistema autónomo que permite detectar anomalías acústicas en máquinas industriales mediante sensores conectados a dispositivos Raspberry Pi.
El sistema procesa el audio en tiempo real, genera alertas cuando detecta comportamientos anómalos y ofrece una interfaz web para la supervisión técnica.

---

## Características principales

- **Captación de sonido**: Raspberry Pi + sensor Adafruit I2S MEMS, graban segmentos de audio de la maquinaria.
- **Procesamiento de audio con IA**:
Los audios se convierten a espectogramas y se analizan con:
  - Autoencoder (detección no supervisada de anomalías).
  - Clasificador supervisado (identificación del tipo de fallo).
- **Comunicación mediante MQTT**:envío eficiente de alertas desde el dispositivo a servidor.
- **Base de datos SQL**: almacenamiento de máquinas, alertas y audios asociados.
- **Interfaz web** (React + Streamlit): visualización de alertas, gráficas sonoras e interacción con los técnicos.
- **Notificaciones**: envío de alertas por correo electrónico.
- **Aprendizaje continuo**: reentrenamiento del modelo con ejemplos etiquetados por técnicos.

---

## Flujo del proyecto

![Flujo del proyecto](/imagen_de_marca/flujo.png)
---

## Estructura del repo

CuruxIA/   
├── docu/               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Documentación extensa, elevator pitch y demo del proyecto    
├── estructura/         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Base de datos, pub/sub y frontend   
├── imagen_de_marca/    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Branding y logo del proyecto   
├── modelosIA/          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Scripts de entrenamiento y predicción   
├── requirements.txt    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Dependencias principales   
├── .env.example        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Variables de entorno de ejemplo   
└── README.md           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Este archivo   

## Instalación y despliegue

**1. Crear entorno virtual e instalar dependencias**   
Copia el archivo .env.example y renómbralo a .env.   
```
cp .env.example .env
python -m venv .venv  
source .venv/bin/activate   
#Dependencias del sistema 
sudo apt update && sudo apt install pkg-config libmysqlclient-dev  
#Instalar Python requirements     
pip install mysqlclient   
pip install -r requirements.txt   
```   
**2. Instalar y configurar MySQL**  
```
sudo apt update && sudo apt install mysql-server mysql-client    
sudo mysql  
CREATE DATABASE curuxia_project;  
CREATE USER 'curuxia_admin'@'localhost' IDENTIFIED BY 'clave_segura';  
GRANT ALL PRIVILEGES ON curuxia_project.* TO 'curuxia_admin'@'localhost';  
FLUSH PRIVILEGES;  
EXIT;  
```   
**3. Crear la base de datos e insertar datos de prueba**  
```
cd estructura/database   
python3 create_database.py  
python3 insert_alerts_batch.py  
```  
---

## ▶️ Ejecución 

**Backend (Flask API)**  
```
cd estructura/ui_react/backend  
python3 app.py  

#Prueba la API:    
curl http://127.0.0.1:5000/api/alerts  
```  
**Frontend (React)**  
```
cd estructura/ui_react/web   
npm install  
npm install leaflet  
npm start  
```  
**Dashboard Streamlit (interfaz inicial)**  
```
cd estructura/ui_react  
streamlit run main.py  
```  
**Pub/Sub con MQTT**  
En el ordenador local (subscriber): 
```
cd estructura/pub_sub  
python3 subscriber.py  
```  
---

## Equipo

- **Martín Amor** - [LinkedIn](https://www.linkedin.com/in/mart%C3%ADn-amor-garc%C3%ADa/)
- **Celia Incera** - [LinkedIn](https://www.linkedin.com/in/celia-incera-alonso-b45b9a2b8?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
- **Inés Poses** - [LinkedIn](https://www.linkedin.com/in/in%C3%A9s-poses-gonz%C3%A1lez-73a80a299/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- **Noelia Sánchez** - [LinkedIn](https://www.linkedin.com/in/noelia-sanchez-dominguez)
