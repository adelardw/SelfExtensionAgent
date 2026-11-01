"""Verified SCRIPTED demonstrations for SFT; never on-policy RL rollouts."""
import json
from pathlib import Path

from .episodes import EpisodeEnvironment, SYSTEM, grade_project, project_tasks


def export_demonstrations(destination, count=24, seed=3812):
    if not 1 <= count <= 200: raise ValueError('Use 1..200 demonstration tasks')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    rows, provenance = [], []
    for task in project_tasks(seed, count, 'train'):
        a, b = task.problem.parameters
        expr = f'min(max(x, {a}), {b})' if task.problem.family == 'clamp' else f'{a}*x+({b})'
        actions = [
            {'action': 'tool', 'tool': 'search', 'args': {'pattern': 'def transform'}},
            {'action': 'tool', 'tool': 'read', 'args': {'path': task.target}},
            {'action': 'tool', 'tool': 'edit', 'args': {'path': task.target, 'old': 'return x', 'new': 'return '+expr}},
            {'action': 'tool', 'tool': 'check', 'args': {'name': 'tests'}},
            {'action': 'finish', 'summary': 'Fixed transform; registered tests pass.'},
        ]
        env = EpisodeEnvironment(task, destination/'projects'/task.id)
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task.prompt}]
        task_rows, observations = [], []
        for action in actions:
            text = json.dumps(action, separators=(',', ':'))
            observation = env.apply(text)
            if not observation['ok']: raise ValueError('Scripted demonstration failed: '+str(observation))
            task_rows.append({'messages': messages + [{'role': 'assistant', 'content': text}]})
            observations.append({'action': action, 'observation': observation})
            messages = messages + [{'role': 'assistant', 'content': text},
                {'role': 'user', 'content': 'Tool observation: '+json.dumps(observation)}]
        grade = grade_project(task, env.snapshot())
        if not env.done or not grade['correct']: raise ValueError('Demonstration failed independent grading')
        rows.extend(task_rows)
        provenance.append({'task_id': task.id, 'family': task.problem.family, 'split': 'train',
            'source': 'scripted_expert', 'grpo_eligible': False, 'grade': grade, 'steps': observations})
    report = {'source': 'scripted_expert', 'grpo_ready': False, 'tasks': len(provenance),
              'rows': len(rows), 'seed': seed, 'heldout_family_excluded': 'distance', 'episodes': provenance}
    for name, text in [('train.jsonl', ''.join(json.dumps(r)+'\n' for r in rows)),
                       ('demonstrations.json', json.dumps(report, indent=2)+'\n')]:
        path = destination/name; path.write_text(text); path.chmod(0o600)
    return report
