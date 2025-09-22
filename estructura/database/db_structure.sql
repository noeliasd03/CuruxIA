USE curuxia_project;

CREATE TABLE machine (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    public_id VARCHAR(50) UNIQUE, 
    place VARCHAR(100), 
    machine_type VARCHAR(100), 
    power INT
);

CREATE TABLE alert (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    machine_id INT, 
    date_time DATETIME, 
    alert_type VARCHAR(100), 
    audio_record LONGTEXT, 
    estado VARCHAR(50) DEFAULT 'Pendiente',
    FOREIGN KEY (machine_id) REFERENCES machine(id) ON DELETE CASCADE
);
