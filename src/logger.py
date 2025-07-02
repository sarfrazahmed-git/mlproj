import logging
import os
from datetime import datetime

Log_file = f"{datetime.now().strftime('%Y-%m-%d')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)
Log_file_path = os.path.join(log_path, Log_file)

logging.basicConfig(
    filename=Log_file_path,
    level=logging.INFO,
    format="[ %(asctime)s ] - %(lineno)d - %(levelname)s - %(message)s"
)
