"""
Prediction History Logger
Stores lightweight API prediction metadata to CSV for traceability.
"""

import csv
import os
from datetime import datetime


class PredictionLogger:
    """Append prediction request outcomes to a CSV log file."""

    def __init__(self, output_dir='prediction_logs', filename='predictions.csv'):
        self.output_dir = output_dir
        self.filename = filename
        os.makedirs(self.output_dir, exist_ok=True)
        self.filepath = os.path.join(self.output_dir, self.filename)
        self._ensure_header()

    def _ensure_header(self):
        if os.path.exists(self.filepath):
            return

        with open(self.filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp',
                'endpoint',
                'status',
                'model',
                'prediction',
                'confidence',
                'error'
            ])

    def log(self, endpoint, status, model=None, prediction=None, confidence=None, error=None):
        row = [
            datetime.now().isoformat(timespec='seconds'),
            endpoint,
            status,
            model or '',
            prediction or '',
            '' if confidence is None else round(float(confidence), 6),
            error or ''
        ]

        with open(self.filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def get_log_path(self):
        return self.filepath
