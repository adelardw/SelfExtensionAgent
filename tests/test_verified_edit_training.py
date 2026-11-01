import json
import pytest

from src.agent_train.verified_edits import EditTask, task_set


def completion(expr):
    return json.dumps({'action': 'tool', 'tool': 'edit', 'args': {
        'path': 'value.py', 'old': 'return x', 'new': 'return ' + expr}})


def test_reward_checks_behavior_not_spelling_or_self_report():
    task = EditTask('clamp', '', (-3, 4), 'train')
    assert task.verify(completion('min(max(x, -3), 4)')) == 1
    assert task.verify(completion('-3 if x < -3 else 4 if x > 4 else x')) == 1
    assert task.verify(completion('x')) == 0
    assert task.verify('{"action":"finish","summary":"All tests passed"}') == 0


@pytest.mark.parametrize('expr', ["__import__('os').system('touch /tmp/not-allowed')", 'x.__class__',
    '[x for x in range(1000000)]', '10 ** 10000000', '9999999999999999', 'x / 1', 'True', 'min(x)', '1 // 0'])
def test_generated_code_has_no_execution_or_unbounded_expression_path(expr):
    assert EditTask('affine', '', (1, 0), 'train').verify(completion(expr)) == 0


def test_tool_boundary_and_json_are_part_of_reward():
    task = EditTask('affine', '', (2, 3), 'train')
    assert task.verify(completion('2*x+3')) == 1
    for field, value in [('path', '../escape.py'), ('old', 'anything')]:
        action = json.loads(completion('2*x+3')); action['args'][field] = value
        assert task.verify(json.dumps(action)) == 0
    assert task.verify('```json\n' + completion('2*x+3') + '\n```') == 0


def test_train_test_task_families_are_disjoint_and_reproducible():
    train, test = task_set(), task_set(split='test')
    assert not {t.family for t in train} & {t.family for t in test}
    assert train == task_set() and test == task_set(split='test')
    assert all(t.verify(completion(f'abs(x-({t.parameters[0]}))')) == 1 for t in test)
