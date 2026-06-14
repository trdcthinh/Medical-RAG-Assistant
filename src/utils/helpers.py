# -*- coding: utf-8 -*-
import yaml
import os
import logging

def load_config(config_path="config.yaml"):
    """Đọc tệp tin cấu hình YAML"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_env(env_path=".env"):
    """Đọc và nạp các biến môi trường từ file .env vào OS"""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

def setup_logging(log_path="logs/app.log"):
    """Thiết lập logging giám sát hệ thống (Tiêu chuẩn DataOps Reliability)"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8"
    )
    return logging.getLogger("MedicalRAG")