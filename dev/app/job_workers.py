"""Single-process database-backed workers for the presentation deployment."""

import asyncio
import logging
from pathlib import Path

from app.config import settings
from app.data_service import (
    block_recommendation_job,
    claim_analysis_job,
    claim_recommendation_job,
    complete_analysis_job,
    complete_recommendation_job,
    create_or_get_recommendation_job,
    fail_recommendation_job,
    mark_analysis_failed,
    recover_stale_jobs,
    set_analysis_stage,
)
from app.image_utils import load_and_validate
from app.pipeline import run_pipeline
from app.recommendation_service import AllRecommendationStrategiesFailed, recommend_with_fallback

logger = logging.getLogger(__name__)

_tasks: list[asyncio.Task] = []
_stopping = asyncio.Event()
_local_inference_semaphore = asyncio.Semaphore(1)


async def _process_analysis(job: dict) -> None:
    analysis_id = job["id"]
    local_slot_acquired = False
    local_slot_released = False

    def advance_stage(stage: str) -> None:
        nonlocal local_slot_released
        set_analysis_stage(analysis_id, stage)
        if stage == "analyzing_regions" and not local_slot_released:
            _local_inference_semaphore.release()
            local_slot_released = True

    try:
        contents = await asyncio.to_thread(Path(job["input_image_path"]).read_bytes)
        image = await asyncio.to_thread(load_and_validate, contents)
        await _local_inference_semaphore.acquire()
        local_slot_acquired = True
        result = await run_pipeline(
            image,
            lang=job.get("lang") or "en",
            excluded_allergens=list(job.get("excluded_allergens") or []),
            stage_callback=advance_stage,
        )
        rec = create_or_get_recommendation_job(
            analysis_id,
            list(job.get("excluded_allergens") or []),
            job.get("lang") or "en",
        )
        if result.get("recommendations_blocked"):
            block_recommendation_job(rec["id"])
        complete_analysis_job(analysis_id, result)
        logger.info("Background analysis completed (id=%s)", analysis_id)
    except Exception as exc:
        logger.exception("Background analysis failed (id=%s)", analysis_id)
        mark_analysis_failed(analysis_id, str(exc))
    finally:
        if local_slot_acquired and not local_slot_released:
            _local_inference_semaphore.release()


async def _analysis_worker(worker_number: int) -> None:
    while not _stopping.is_set():
        job = await asyncio.to_thread(claim_analysis_job)
        if job is None:
            await asyncio.sleep(settings.job_poll_interval_seconds)
            continue
        logger.info("Analysis worker %d claimed %s", worker_number, job["id"])
        await _process_analysis(job)


async def _process_recommendation(job: dict) -> None:
    result_json = dict(job.get("result_json") or {})
    try:
        result = await recommend_with_fallback(
            fitzpatrick=result_json["tom_geral_fitzpatrick"],
            undertone=result_json["subtom_predominante"],
            skin_hex=result_json.get("tom_geral_hex") or "#c68b6e",
            condition_map=result_json.get("segformer_condition_map") or result_json.get("condition_map"),
            excluded_allergens=list(job.get("excluded_allergens") or []),
            lang=job.get("lang") or result_json.get("lang", "en"),
            force_fallback=bool(job.get("force_fallback")),
        )
        complete_recommendation_job(job["id"], result)
        logger.info("Recommendation completed (id=%s)", job["id"])
    except AllRecommendationStrategiesFailed as exc:
        fail_recommendation_job(job["id"], str(exc))
    except Exception as exc:
        logger.exception("Recommendation worker failed (id=%s)", job["id"])
        fail_recommendation_job(job["id"], str(exc))


async def _recommendation_worker(worker_number: int) -> None:
    while not _stopping.is_set():
        job = await asyncio.to_thread(claim_recommendation_job)
        if job is None:
            await asyncio.sleep(settings.job_poll_interval_seconds)
            continue
        logger.info("Recommendation worker %d claimed %s", worker_number, job["id"])
        await _process_recommendation(job)


async def _recovery_worker() -> None:
    while not _stopping.is_set():
        await asyncio.sleep(60)
        recovered = await asyncio.to_thread(recover_stale_jobs, settings.job_stale_after_seconds)
        if any(recovered):
            logger.warning("Recovered stale jobs (analyses=%d recommendations=%d)", *recovered)


async def start_workers() -> None:
    _stopping.clear()
    recovered = await asyncio.to_thread(recover_stale_jobs, settings.job_stale_after_seconds)
    if any(recovered):
        logger.warning("Recovered stale jobs (analyses=%d recommendations=%d)", *recovered)
    for index in range(settings.analysis_worker_concurrency):
        _tasks.append(asyncio.create_task(_analysis_worker(index + 1), name=f"analysis-worker-{index + 1}"))
    for index in range(settings.recommendation_worker_concurrency):
        _tasks.append(asyncio.create_task(_recommendation_worker(index + 1), name=f"recommendation-worker-{index + 1}"))
    _tasks.append(asyncio.create_task(_recovery_worker(), name="job-recovery-worker"))


async def stop_workers() -> None:
    _stopping.set()
    for task in _tasks:
        task.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)
    _tasks.clear()
