"""Audit traces and export only explicitly training-eligible, externally checked runs."""
import json
from collections import Counter
from pathlib import Path

from .trajectory import read_events


def inspect_trace(path):
    events = read_events(path)
    start = next((e['data'] for e in events if e['kind'] == 'run_start'), {})
    outcomes = [e['data'] for e in events if e['kind'] == 'outcome']
    evaluations = [e['data'] for e in events if e['kind'] == 'evaluation']
    evaluation = evaluations[-1] if evaluations else {}
    reasons = []
    if not events or events[-1]['kind'] != 'evaluation': reasons.append('unsealed_trace')
    if not outcomes: reasons.append('missing_outcome')
    if not evaluations: reasons.append('no_external_verifier')
    if evaluation.get('split') != 'train': reasons.append('not_training_split')
    if evaluation.get('training_eligible') is not True: reasons.append('not_eligible')
    if evaluation.get('reward') != 1.0: reasons.append('not_verified_success')
    if any(e['redactions'] for e in events): reasons.append('redacted')
    if outcomes and outcomes[-1].get('status') != 'done': reasons.append('not_finished')
    requests, responses, validated, executed, finishes = {}, {}, {}, set(), set()
    for event in events:
        key = event['call_id']
        if event['kind'] == 'request': requests[key] = event['data']
        elif event['kind'] == 'response': responses[key] = event['data']
        elif event['kind'] == 'validated': validated[key] = event['data']
        elif event['kind'] == 'action' and event['data']['ok']: executed.add(key)
        elif event['kind'] == 'finish' and event['data']['status'] == 'done': finishes.add(key)
    if set(requests) - set(responses): reasons.append('incomplete_model_calls')
    executed |= {key for key in finishes if validated.get(key, {}).get('action') == 'finish'}
    rows = []
    skipped = Counter()
    for key in requests:
        if key not in validated or key not in executed: continue
        request, response = requests.get(key, {}), responses.get(key, {})
        messages = request.get('messages', [])
        if request.get('generation', {}).get('tools') or response.get('tool_calls'):
            skipped['native_tool_template_required'] += 1
            continue
        if not request.get('schema') or not messages or not response.get('content'):
            skipped['missing_exact_request_response'] += 1
            continue
        if response.get('finish_reason') == 'length':
            skipped['truncated_response'] += 1
            continue
        # MLX chat SFT with --mask-prompt trains only this final assistant message.
        rows.append({'messages': messages + [{'role': 'assistant', 'content': response['content']}]})
    return {'path': str(path), 'model': start.get('model'), 'harness_id': start.get('harness', {}).get('source_id'),
            'family': evaluation.get('family'), 'split': evaluation.get('split'),
            'reward': evaluation.get('reward'), 'eligible': not reasons, 'excluded': reasons,
            'requests': len(requests), 'responses': len(responses), 'candidate_sft_steps': len(rows),
            'skipped_steps': dict(skipped), 'rows': rows if not reasons else []}


def export(paths, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    audits = []
    rows = []
    seen = set()
    for path in sorted(set(map(str, paths))):
        try: audit = inspect_trace(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            audits.append({'path': path, 'eligible': False, 'excluded': ['unreadable_or_corrupt'], 'error': str(exc)})
            continue
        for row in audit.pop('rows'):
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if serialized not in seen:
                rows.append(serialized); seen.add(serialized)
        audits.append(audit)
    # Do not invent a random per-step validation split: siblings from one task leak.
    # A separate held-out task-family dataset must be supplied for model evaluation.
    (destination / 'train.jsonl').write_text(''.join(row + '\n' for row in rows))
    report = {'traces': len(audits), 'sft_rows': len(rows), 'audits': audits,
              'grpo_ready': False,
              'note': 'Historical cloud traces are not on-policy GRPO batches. Collect fresh local groups with token IDs and policy versions. No validation split is fabricated.'}
    (destination / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (destination / 'train.jsonl').chmod(0o600)
    (destination / 'audit.json').chmod(0o600)
    return report
