"""
Background Training Job Manager for Zemo.

Provides an asynchronous, non-blocking background execution architecture
for model training with real-time status tracking and persistent logs.
Designed to cleanly support Celery / Redis workers if scale requires later.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from sqlalchemy import select, update

try:
    from core.database.config import Database, get_database
    from core.database.models.experiment import Experiment, ExperimentStatus
    from services.ml.pipeline_builder import TrainingConfig
    from services.ml.trainer import MLTrainingEngine, TrainingResult
except ImportError:
    from backend.core.database.config import Database, get_database
    from backend.core.database.models.experiment import Experiment, ExperimentStatus
    from backend.services.ml.pipeline_builder import TrainingConfig
    from backend.services.ml.trainer import MLTrainingEngine, TrainingResult

logger = logging.getLogger(__name__)


class ActiveJobState:
    """In-memory active state of a running training task."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.status = ExperimentStatus.PENDING.value
        self.logs: List[Dict[str, Any]] = []
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.result: Optional[TrainingResult] = None
        self.task: Optional[asyncio.Task] = None
        self.is_cancelled = False

    def add_log(self, level: str, message: str) -> None:
        self.logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.upper(),
            "message": message,
        })


class TrainingJobManager:
    """
    Central manager for initiating, tracking, and canceling training jobs.
    Uses asyncio background tasks running on worker threads via asyncio.to_thread
    to ensure the FastAPI event loop is NEVER blocked by Scikit-learn computations.
    """

    _instance: Optional["TrainingJobManager"] = None

    def __init__(
        self,
        training_engine: Optional[MLTrainingEngine] = None,
        db_getter: Optional[Callable[[], Database]] = None,
    ):
        self.engine = training_engine or MLTrainingEngine()
        self.get_db = db_getter or get_database
        self._jobs: Dict[str, ActiveJobState] = {}

    @classmethod
    def get_instance(cls) -> "TrainingJobManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_job_state(self, experiment_id: str) -> Optional[ActiveJobState]:
        return self._jobs.get(experiment_id)

    def cancel_job(self, experiment_id: str) -> bool:
        job = self._jobs.get(experiment_id)
        if job and job.status in (ExperimentStatus.PENDING.value, ExperimentStatus.RUNNING.value):
            job.is_cancelled = True
            job.status = ExperimentStatus.CANCELLED.value
            job.add_log("warning", "Experiment training cancelled by user request.")
            if job.task and not job.task.done():
                job.task.cancel()
            return True
        return False

    async def launch_job(
        self,
        experiment_id: str,
        df: pd.DataFrame,
        config: TrainingConfig,
    ) -> ActiveJobState:
        """
        Creates in-memory tracker and schedules background task.
        Returns immediately without blocking the caller.
        """
        job = ActiveJobState(experiment_id)
        self._jobs[experiment_id] = job

        # Launch async task in background
        task = asyncio.create_task(self._execute_training_job(job, df, config))
        job.task = task
        return job

    async def _execute_training_job(
        self,
        job: ActiveJobState,
        df: pd.DataFrame,
        config: TrainingConfig,
    ) -> None:
        """
        Worker routine that manages database state transitions and executes CPU training.
        """
        experiment_id = job.experiment_id
        db = self.get_db()

        # 1. Update status to RUNNING
        job.status = ExperimentStatus.RUNNING.value
        job.started_at = datetime.utcnow()
        job.add_log("info", "Background training job started.")

        try:
            async with db.session() as session:
                await session.execute(
                    update(Experiment)
                    .where(Experiment.id == experiment_id)
                    .values(
                        status=ExperimentStatus.RUNNING.value,
                        started_at=job.started_at,
                        logs=job.logs,
                    )
                )
        except Exception as e:
            logger.warning(f"Could not update database status to RUNNING for {experiment_id}: {e}")

        # 2. Execute Training in separate thread to avoid blocking event loop
        try:
            # Check for cancellation before expensive computation
            if job.is_cancelled:
                return

            def log_callback(level: str, msg: str) -> None:
                job.add_log(level, msg)

            # asyncio.to_thread runs CPU-bound Scikit-learn training in default thread pool
            result: TrainingResult = await asyncio.to_thread(
                self.engine.train,
                experiment_id=experiment_id,
                df=df,
                config=config,
                log_callback=log_callback,
            )

            if job.is_cancelled:
                job.status = ExperimentStatus.CANCELLED.value
                job.completed_at = datetime.utcnow()
                job.add_log("warning", "Experiment cancelled after training step.")
                async with db.session() as session:
                    await session.execute(
                        update(Experiment)
                        .where(Experiment.id == experiment_id)
                        .values(
                            status=ExperimentStatus.CANCELLED.value,
                            completed_at=job.completed_at,
                            logs=job.logs,
                        )
                    )
                return

            # 3. Mark COMPLETED
            job.status = ExperimentStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            job.result = result

            async with db.session() as session:
                await session.execute(
                    update(Experiment)
                    .where(Experiment.id == experiment_id)
                    .values(
                        status=ExperimentStatus.COMPLETED.value,
                        completed_at=job.completed_at,
                        metrics=result.metrics,
                        artifact_path=result.artifact_path,
                        model_size_bytes=result.model_size_bytes,
                        duration_seconds=result.duration_seconds,
                        hyperparameters=result.hyperparameters,
                        logs=job.logs,
                    )
                )
            logger.info(f"Experiment {experiment_id} training completed and persisted.")

        except asyncio.CancelledError:
            job.status = ExperimentStatus.CANCELLED.value
            job.completed_at = datetime.utcnow()
            job.add_log("warning", "Training task received cancellation signal.")
            try:
                async with db.session() as session:
                    await session.execute(
                        update(Experiment)
                        .where(Experiment.id == experiment_id)
                        .values(
                            status=ExperimentStatus.CANCELLED.value,
                            completed_at=job.completed_at,
                            logs=job.logs,
                        )
                    )
            except Exception:
                pass

        except Exception as exc:
            err_msg = str(exc)
            logger.error(f"Experiment {experiment_id} failed: {err_msg}", exc_info=True)
            job.status = ExperimentStatus.FAILED.value
            job.completed_at = datetime.utcnow()
            job.error_message = err_msg
            job.add_log("error", f"Training failed with error: {err_msg}")

            try:
                async with db.session() as session:
                    await session.execute(
                        update(Experiment)
                        .where(Experiment.id == experiment_id)
                        .values(
                            status=ExperimentStatus.FAILED.value,
                            completed_at=job.completed_at,
                            error_message=err_msg,
                            logs=job.logs,
                        )
                    )
            except Exception as db_err:
                logger.error(f"Failed to record failure state in database: {db_err}")
