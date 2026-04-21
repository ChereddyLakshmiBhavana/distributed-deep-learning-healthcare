"""
Distributed Worker
Polls the SQLite job queue and processes prediction jobs.
"""

import json
import time

from distributed_engine import DistributedEngine
from model_loader import ModelLoader
from prediction_service import predict_from_image_bytes, predict_from_base64


def process_job(job, model_loader, engine):
    payload = job['payload']
    task_type = job['task_type']

    if task_type == 'predict_base64':
        return predict_from_base64(
            model_loader=model_loader,
            image_data=payload['image'],
            model_name=payload.get('model', 'random_forest_model'),
            validate=payload.get('validate', True)
        )

    if task_type == 'predict_bytes':
        image_bytes = bytes.fromhex(payload['image_hex'])
        return predict_from_image_bytes(
            model_loader=model_loader,
            image_bytes=image_bytes,
            model_name=payload.get('model', 'random_forest_model'),
            validate=payload.get('validate', True)
        )

    if task_type == 'federated_aggregate':
        return engine.aggregate_federated_updates(
            model_name=payload['model_name'],
            round_id=payload.get('round_id')
        )

    raise ValueError(f'Unsupported task type: {task_type}')


def main():
    model_loader = ModelLoader()
    model_loader.load_classical_models()
    model_loader.load_cnn_model()
    model_loader.load_fast_resnet_model()

    engine = DistributedEngine()
    print('Distributed worker started. Polling the queue...')

    while True:
        job = engine.claim_next_job()
        if not job:
            time.sleep(2)
            continue

        try:
            result = process_job(job, model_loader, engine)
            engine.update_job(job['id'], 'completed', result=result)
            print(f"✓ Completed job {job['id']} ({job['task_type']})")
        except Exception as exc:
            engine.update_job(job['id'], 'failed', error=str(exc))
            print(f"⚠ Job {job['id']} failed: {exc}")


if __name__ == '__main__':
    main()
