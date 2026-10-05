"""Bind an already captured RunPod episode; never open paths supplied by a model."""
import hashlib

from .contract import ContractError, finite, snapshot
from .runtime import extract_plan


def validate_episode(record, evidence, artifacts, expected, *, now_unix, max_age_s):
    """Artifacts are host-loaded bytes; expected is a trusted request manifest.

    An oriented VLM image and original JPEG may have different hashes. The host
    records the actual sent image digest; this function never rotates an image.
    Captured Unix time is checked separately from runtime monotonic timestamps.
    """
    record, evidence, expected = snapshot(record), snapshot(evidence), snapshot(expected)
    try:
        if evidence['schema'] != 'scene2action-fresh-episode-evidence-v1':
            raise ContractError('EPISODE_SCHEMA_UNSUPPORTED')
        for k in ('case_id', 'model', 'reset_count'):
            if type(evidence[k]) is not type(expected[k]) or evidence[k] != expected[k]:
                raise ContractError('EPISODE_IDENTITY_MISMATCH')
        if record['case_id'] != expected['case_id'] or record['model_label'] != expected['model']:
            raise ContractError('VLM_REQUEST_IDENTITY_MISMATCH')
        if (not finite(now_unix) or not finite(max_age_s) or max_age_s <= 0
                or not finite(evidence['captured_at_unix'])
                or not 0 <= now_unix - evidence['captured_at_unix'] <= max_age_s):
            raise ContractError('EPISODE_STALE_OR_FUTURE')
        if evidence['ground_truth_used_for_control'] is not False:
            raise ContractError('GROUND_TRUTH_CONTROL_NOT_ALLOWED')
        if type(evidence['reset_count']) is not int or evidence['reset_count'] < 0:
            raise ContractError('INVALID_RESET_COUNT')
        if (not finite(evidence['reset_completed_at_unix'])
                or evidence['reset_completed_at_unix'] > evidence['captured_at_unix']):
            raise ContractError('INVALID_CAPTURE_ORDER')
        if set(artifacts) != {'rgb', 'depth', 'camera_status'}:
            raise ContractError('EPISODE_ARTIFACTS_REQUIRED')
        for name, value in artifacts.items():
            if not isinstance(value, bytes) or hashlib.sha256(value).hexdigest() != evidence[name + '_sha256']:
                raise ContractError('EPISODE_ARTIFACT_MISMATCH')
        source = record['input']
        if source['episode_evidence'] != evidence or source['same_episode_camera_for_vlm'] is not True:
            raise ContractError('VLM_EPISODE_MISMATCH')
        if source['raw_image_sha256'] != evidence['rgb_sha256']:
            raise ContractError('RAW_IMAGE_MISMATCH')
        for source_key, expected_key in [('image_sha256', 'sent_image_sha256'),
                                         ('camera_orientation_transform', 'orientation_transform'),
                                         ('request_sha256', 'request_sha256')]:
            if source[source_key] != expected[expected_key]:
                raise ContractError('SENT_REQUEST_MISMATCH')
        return extract_plan(record)
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError('EPISODE_EVIDENCE_INCOMPLETE') from exc
