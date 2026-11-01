import json

import pytest

from src.agent_train.config import DEFAULT_CONFIG, load_config
from src.agent_train.episodes import (
    ActionSample, ContextLimit, EpisodeEnvironment, collect_episode, episode_credit,
    grade_project, project_tasks,
)


def action(tool, **args):
    return json.dumps({'action': 'tool', 'tool': tool, 'args': args})


FINISH = '{"action":"finish","summary":"done"}'


def solution(task):
    a = task.problem.parameters
    if task.problem.family == 'clamp': return f'min(max(x, {a[0]}), {a[1]})'
    if task.problem.family == 'affine': return f'{a[0]}*x+({a[1]})'
    return f'abs(x-({a[0]}))'


def edit(task, expr=None):
    return action('edit', path=task.target, old='return x', new='return '+(expr or solution(task)))


def scripted(actions):
    pending = iter(actions)
    def generate(messages):
        text = next(pending)
        # Deliberately distinct context/action IDs for loss-boundary testing.
        return ActionSample((101, 102, len(messages)), (201, 202), (-1., -2.), text)
    return generate


def test_end_to_end_tool_sequence_uses_actual_observations_and_external_reward(tmp_path):
    task = project_tasks(812, 1, 'train')[0]
    actions = [action('search', pattern='def transform'), action('read', path=task.target),
               edit(task), action('check', name='tests'), FINISH]
    episode = collect_episode(task, scripted(actions), tmp_path/'repo')
    assert episode['reward'] == 1 and episode['finished'] and episode['training_eligible']
    assert len(episode['steps']) == 5
    observed = episode['steps'][0]['observation']['output']
    assert task.target in observed
    assert observed in episode['steps'][1]['messages'][-1]['content']
    assert 'return '+solution(task) in episode['final_files'][task.target]
    assert episode['grade']['verification_domain'] == 'all integers [-20,20]'


def test_failed_check_can_be_observed_and_repaired_in_same_episode(tmp_path):
    task = project_tasks(812, 2, 'train')[1]
    episode = collect_episode(task, scripted([action('check', name='tests'), edit(task),
        action('check', name='tests'), FINISH]), tmp_path/'repo')
    assert episode['steps'][0]['observation']['ok'] is False
    assert episode['reward'] == 1


def test_checks_and_self_report_do_not_replace_independent_grading(tmp_path):
    task = project_tasks(812, 1, 'train')[0]
    # Deliberately overfit all public inputs. The private finite domain catches it.
    correct = solution(task)
    spoof = f'({correct}) if x == -2 else ({correct}) if x == 0 else ({correct}) if x == 2 else 999'
    episode = collect_episode(task, scripted([edit(task, spoof), action('check', name='tests'), FINISH]), tmp_path/'repo')
    assert episode['steps'][1]['observation']['ok']
    assert episode['finished'] and episode['reward'] == 0 and not episode['grade']['correct']


def test_early_finish_and_stale_success_do_not_earn_reward(tmp_path):
    task = project_tasks(812, 1, 'train')[0]
    env = EpisodeEnvironment(task, tmp_path/'repo')
    assert not env.apply(FINISH)['ok']
    assert env.apply(edit(task))['ok']
    assert env.apply(action('check', name='tests'))['ok']
    assert env.apply(action('edit', path=task.target, old='return '+solution(task), new='return 0'))['ok']
    assert not env.apply(FINISH)['ok'] and not env.done


@pytest.mark.parametrize('bad', [
    'not json', '[]', '{"action":"tool","tool":[],"args":{}}',
    action('run', argv=['rm', '-rf', '/']), action('read', path='/etc/passwd'),
    action('read', path='../secret'), action('edit', path='tests/cases.json', old='[', new='[]'),
    action('check', name='invented'), action('search', pattern=''),
])
def test_action_boundary_rejects_unsafe_or_unavailable_operations(tmp_path, bad):
    task = project_tasks(812, 1, 'train')[0]
    env = EpisodeEnvironment(task, tmp_path/'repo')
    before = env.snapshot()
    assert env.apply(bad)['ok'] is False
    assert env.snapshot() == before


def test_generated_python_is_never_executed(tmp_path):
    task = project_tasks(812, 1, 'train')[0]
    marker = tmp_path/'never-created'
    malicious = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    env = EpisodeEnvironment(task, tmp_path/'repo')
    assert env.apply(edit(task, malicious))['ok']
    assert not env.apply(action('check', name='tests'))['ok']
    assert not grade_project(task, env.snapshot())['correct']
    assert not marker.exists()


def test_candidate_projects_are_isolated_and_unrelated_edits_fail_grading(tmp_path):
    task = project_tasks(812, 1, 'train')[0]
    a, b = EpisodeEnvironment(task, tmp_path/'a'), EpisodeEnvironment(task, tmp_path/'b')
    a.apply(edit(task))
    assert a.snapshot() != b.snapshot() and b.snapshot() == task.files
    other = next(p for p in task.files if p.endswith('.py') and p != task.target)
    a.apply(action('edit', path=other, old='return x + 1', new='return 0'))
    assert not grade_project(task, a.snapshot())['correct']


def test_symlink_cannot_escape_project(tmp_path):
    task = project_tasks(812, 1, 'train')[0]
    env = EpisodeEnvironment(task, tmp_path/'repo')
    outside = tmp_path/'outside'; outside.write_text('secret')
    path = env.path(task.target); path.unlink(); path.symlink_to(outside)
    assert not env.apply(action('read', path=task.target))['ok']
    assert not env.apply(edit(task))['ok']
    assert outside.read_text() == 'secret'


def test_time_interruption_is_not_a_training_failure_and_infrastructure_errors_propagate(tmp_path):
    task = project_tasks(812, 1, 'train')[0]
    result = collect_episode(task, scripted([]), tmp_path/'cancelled', abort=lambda: True)
    assert result['reward'] is None and not result['training_eligible']
    def broken(messages): raise RuntimeError('backend failed')
    with pytest.raises(RuntimeError, match='backend failed'):
        collect_episode(task, broken, tmp_path/'broken')


def test_context_limit_retains_actions_and_counts_as_bounded_attempt(tmp_path):
    task = project_tasks(812, 1, 'train')[0]
    calls = 0
    def generate(messages):
        nonlocal calls
        calls += 1
        if calls == 2: raise ContextLimit()
        return ActionSample((1,), (2,), (-.5,), action('tree'))
    result = collect_episode(task, generate, tmp_path/'repo')
    assert result['stop_reason'] == 'context_limit' and result['reward'] == 0
    assert result['training_eligible'] and len(result['steps']) == 1


def test_group_credit_is_normalized_per_episode_not_per_action():
    episodes = [{'training_eligible': True, 'reward': 1., 'steps': ['a']},
                {'training_eligible': True, 'reward': 0., 'steps': ['b', 'c', 'd']}]
    for e in episodes: e.update(task_id='a', split='train')
    credit = episode_credit(episodes)
    assert [x[2] for x in credit] == pytest.approx([1., -1., -1., -1.], abs=1e-5)
    episodes[0]['reward'] = 0.
    assert all(a == 0 for _, _, a in episode_credit(episodes))
    episodes[0]['training_eligible'] = False
    with pytest.raises(ValueError, match='complete group'): episode_credit(episodes)


def test_credit_rejects_mixed_tasks_or_heldout():
    first = {'training_eligible': True, 'reward': 1., 'steps': ['a'], 'task_id': 'a', 'split': 'train'}
    second = first | {'task_id': 'b'}
    with pytest.raises(ValueError, match='one train task'): episode_credit([first, second])
    with pytest.raises(ValueError, match='one train task'): episode_credit([first, first | {'split': 'test'}])


def test_heldout_families_are_excluded_from_training(tmp_path):
    train, heldout = project_tasks(812, 8, 'train'), project_tasks(813, 3, 'test')
    assert not {t.problem.family for t in train} & {t.problem.family for t in heldout}
    assert train == project_tasks(812, 8, 'train')
    task = heldout[0]
    result = collect_episode(task, scripted([edit(task), action('check', name='tests'), FINISH]), tmp_path/'repo')
    assert result['reward'] == 1 and not result['training_eligible']


def test_profile_and_sampling_policy_validation():
    args = load_config(DEFAULT_CONFIG.with_name('episodes.yml'))
    assert args.reward_kind == 'tool_episodes' and args.episode_max_turns == 8
    assert args.model_name.endswith('Qwen3-0.6B-4bit')
    with pytest.raises(ValueError, match='sampling policy'):
        load_config(DEFAULT_CONFIG.with_name('episodes.yml'), {'temperature': .7})


def test_bootstrap_is_verified_sft_only_and_never_uses_heldout_family(tmp_path):
    from src.agent_train.bootstrap import export_demonstrations
    from src.agent_train.sft import validate_dataset
    output = tmp_path/'demonstrations'
    report = export_demonstrations(output, count=4)
    assert report['rows'] == 20 and not report['grpo_ready']
    assert {e['family'] for e in report['episodes']} == {'clamp', 'affine'}
    assert all(e['grade']['correct'] and not e['grpo_eligible'] and e['source'] == 'scripted_expert' for e in report['episodes'])
    assert validate_dataset(output)['train'] == 20
    with pytest.raises(FileExistsError): export_demonstrations(output)


@pytest.mark.parametrize('tokens,logprobs', [((), ()), ((1,), ()), ((1,), (float('nan'),))])
def test_rollout_requires_real_finite_per_token_logprobs(tokens, logprobs):
    with pytest.raises(ValueError): ActionSample((2,), tokens, logprobs, 'text')
