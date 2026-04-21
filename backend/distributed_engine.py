"""
Distributed Engine
Provides a lightweight distributed engineering layer with:
- job queue
- federated learning registry/aggregation
- distributed runtime summary
"""

from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import torch


class DistributedEngine:
    """SQLite-backed orchestration for queueing and federated aggregation."""

    def __init__(self, state_dir='distributed_state', models_dir='../models'):
        base_dir = Path(__file__).resolve().parent
        state_candidate = Path(state_dir)
        models_candidate = Path(models_dir)

        self.state_dir = (base_dir / state_candidate).resolve() if not state_candidate.is_absolute() else state_candidate
        self.models_dir = (base_dir / models_candidate).resolve() if not models_candidate.is_absolute() else models_candidate
        self.db_path = self.state_dir / 'distributed.db'
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self._initialize_schema()

    def _connect(self):
        return sqlite3.connect(str(self.db_path))

    def _initialize_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS distributed_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS federated_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_name TEXT NOT NULL UNIQUE,
                    site_name TEXT,
                    node_type TEXT NOT NULL DEFAULT 'hospital',
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS federated_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_id TEXT NOT NULL,
                    node_id INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    num_samples INTEGER NOT NULL,
                    metrics_json TEXT,
                    state_blob BLOB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'submitted',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(node_id) REFERENCES federated_nodes(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS distributed_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _now(self):
        return datetime.now().isoformat(timespec='seconds')

    def _log_audit(self, event_type, payload):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO distributed_audit (event_type, payload_json, created_at) VALUES (?, ?, ?)",
                (event_type, json.dumps(payload), self._now())
            )

    def enqueue_job(self, task_type, payload, priority=0):
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO distributed_jobs (task_type, payload_json, status, priority, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?, ?)
                """,
                (task_type, json.dumps(payload), int(priority), self._now(), self._now())
            )
            job_id = cursor.lastrowid

        self._log_audit('job_enqueued', {'job_id': job_id, 'task_type': task_type})
        return job_id

    def list_jobs(self, status=None, limit=50):
        query = "SELECT id, task_type, payload_json, status, priority, created_at, updated_at, result_json, error FROM distributed_jobs"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY priority DESC, id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        jobs = []
        for row in rows:
            jobs.append({
                'id': row[0],
                'task_type': row[1],
                'payload': json.loads(row[2]),
                'status': row[3],
                'priority': row[4],
                'created_at': row[5],
                'updated_at': row[6],
                'result': json.loads(row[7]) if row[7] else None,
                'error': row[8]
            })
        return jobs

    def get_job(self, job_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, task_type, payload_json, status, priority, created_at, updated_at, result_json, error FROM distributed_jobs WHERE id = ?",
                (job_id,)
            ).fetchone()

        if not row:
            return None

        return {
            'id': row[0],
            'task_type': row[1],
            'payload': json.loads(row[2]),
            'status': row[3],
            'priority': row[4],
            'created_at': row[5],
            'updated_at': row[6],
            'result': json.loads(row[7]) if row[7] else None,
            'error': row[8]
        }

    def claim_next_job(self):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM distributed_jobs
                WHERE status = 'pending'
                ORDER BY priority DESC, id ASC
                LIMIT 1
                """
            ).fetchone()

            if not row:
                return None

            job_id = row[0]
            conn.execute(
                "UPDATE distributed_jobs SET status = 'processing', updated_at = ? WHERE id = ?",
                (self._now(), job_id)
            )

        self._log_audit('job_claimed', {'job_id': job_id})
        return self.get_job(job_id)

    def update_job(self, job_id, status, result=None, error=None):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE distributed_jobs
                SET status = ?, result_json = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, json.dumps(result) if result is not None else None, error, self._now(), job_id)
            )

        self._log_audit('job_updated', {'job_id': job_id, 'status': status})

    def register_node(self, node_name, site_name=None, node_type='hospital', metadata=None):
        metadata_json = json.dumps(metadata or {})
        now = self._now()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM federated_nodes WHERE node_name = ?",
                (node_name,)
            ).fetchone()

            if row:
                node_id = row[0]
                conn.execute(
                    """
                    UPDATE federated_nodes
                    SET site_name = ?, node_type = ?, metadata_json = ?, last_seen_at = ?
                    WHERE id = ?
                    """,
                    (site_name, node_type, metadata_json, now, node_id)
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO federated_nodes (node_name, site_name, node_type, metadata_json, created_at, last_seen_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (node_name, site_name, node_type, metadata_json, now, now)
                )
                node_id = cursor.lastrowid

        self._log_audit('node_registered', {'node_id': node_id, 'node_name': node_name})
        return node_id

    def list_nodes(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, node_name, site_name, node_type, metadata_json, created_at, last_seen_at FROM federated_nodes ORDER BY id DESC"
            ).fetchall()

        nodes = []
        for row in rows:
            nodes.append({
                'id': row[0],
                'node_name': row[1],
                'site_name': row[2],
                'node_type': row[3],
                'metadata': json.loads(row[4]) if row[4] else {},
                'created_at': row[5],
                'last_seen_at': row[6]
            })
        return nodes

    def _state_dict_to_blob(self, state_dict):
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    def _blob_to_state_dict(self, blob):
        buffer = io.BytesIO(blob)
        return torch.load(buffer, map_location='cpu')

    def submit_federated_update(self, node_id, model_name, state_dict, num_samples, metrics=None, round_id=None):
        round_id = round_id or f"round_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metrics_json = json.dumps(metrics or {})
        blob = self._state_dict_to_blob(state_dict)
        now = self._now()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO federated_updates (round_id, node_id, model_name, num_samples, metrics_json, state_blob, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'submitted', ?)
                """,
                (round_id, node_id, model_name, int(num_samples), metrics_json, blob, now)
            )
            update_id = cursor.lastrowid

        self._log_audit('federated_update_submitted', {'update_id': update_id, 'round_id': round_id, 'model_name': model_name})
        return update_id, round_id

    def list_federated_updates(self, model_name=None, round_id=None):
        query = "SELECT id, round_id, node_id, model_name, num_samples, metrics_json, state_blob, status, created_at FROM federated_updates"
        clauses = []
        params = []
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if round_id:
            clauses.append("round_id = ?")
            params.append(round_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        updates = []
        for row in rows:
            updates.append({
                'id': row[0],
                'round_id': row[1],
                'node_id': row[2],
                'model_name': row[3],
                'num_samples': row[4],
                'metrics': json.loads(row[5]) if row[5] else {},
                'status': row[7],
                'created_at': row[8]
            })
        return updates

    def aggregate_federated_updates(self, model_name, round_id=None):
        query = "SELECT round_id, node_id, num_samples, metrics_json, state_blob FROM federated_updates WHERE model_name = ? AND status = 'submitted'"
        params = [model_name]
        if round_id:
            query += " AND round_id = ?"
            params.append(round_id)
        query += " ORDER BY id ASC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            raise ValueError(f'No federated updates found for model {model_name}')

        updates = []
        for row in rows:
            updates.append({
                'round_id': row[0],
                'node_id': row[1],
                'num_samples': row[2],
                'metrics': json.loads(row[3]) if row[3] else {},
                'state_dict': self._blob_to_state_dict(row[4])
            })

        target_round = round_id or updates[0]['round_id']
        total_samples = sum(max(1, int(update['num_samples'])) for update in updates)
        aggregated_state = {}

        keys = list(updates[0]['state_dict'].keys())
        for key in keys:
            values = []
            for update in updates:
                tensor = update['state_dict'][key]
                if torch.is_tensor(tensor):
                    values.append(tensor.detach().float() * max(1, int(update['num_samples'])))
                else:
                    values.append(tensor)

            if torch.is_tensor(values[0]):
                summed = values[0]
                for tensor in values[1:]:
                    summed = summed + tensor
                aggregated_state[key] = summed / float(total_samples)
            else:
                aggregated_state[key] = values[0]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.models_dir / f'fedavg_{model_name}_{target_round}_{timestamp}.pth'
        torch.save({
            'model_state_dict': aggregated_state,
            'model_name': model_name,
            'round_id': target_round,
            'num_updates': len(updates),
            'total_samples': total_samples,
            'created_at': self._now()
        }, output_path)

        with self._connect() as conn:
            conn.execute(
                "UPDATE federated_updates SET status = 'aggregated' WHERE model_name = ? AND round_id = ? AND status = 'submitted'",
                (model_name, target_round)
            )

        self._log_audit('federated_round_aggregated', {
            'model_name': model_name,
            'round_id': target_round,
            'num_updates': len(updates),
            'total_samples': total_samples,
            'output_path': str(output_path)
        })

        return {
            'model_name': model_name,
            'round_id': target_round,
            'num_updates': len(updates),
            'total_samples': total_samples,
            'output_path': str(output_path)
        }

    def get_summary(self):
        with self._connect() as conn:
            job_counts = conn.execute(
                "SELECT status, COUNT(*) FROM distributed_jobs GROUP BY status"
            ).fetchall()
            node_count = conn.execute("SELECT COUNT(*) FROM federated_nodes").fetchone()[0]
            update_count = conn.execute("SELECT COUNT(*) FROM federated_updates").fetchone()[0]
            audit_count = conn.execute("SELECT COUNT(*) FROM distributed_audit").fetchone()[0]

        return {
            'state_db': str(self.db_path),
            'jobs': {status: count for status, count in job_counts},
            'nodes_registered': node_count,
            'federated_updates': update_count,
            'audit_events': audit_count,
            'gpu_count': torch.cuda.device_count(),
            'cuda_available': torch.cuda.is_available(),
            'ddp_recommended': torch.cuda.device_count() > 1
        }


def serialize_state_dict(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def deserialize_state_dict(state_dict_b64):
    buffer = io.BytesIO(base64.b64decode(state_dict_b64))
    return torch.load(buffer, map_location='cpu')
