-- Active: 1744941154151@@127.0.0.1@3306
CREATE DATABASE pothole_db;
USE pothole_db;

CREATE TABLE detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp VARCHAR(255),
    image_data LONGBLOB,
    confidence FLOAT
);

TRUNCATE detections