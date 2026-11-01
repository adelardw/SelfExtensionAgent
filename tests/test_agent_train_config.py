import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.agent_train.cli import main
from src.agent_train.config import DEFAULT_CONFIG, describe, load_config
from src.agent_train.preflight import validate_local_model
from src.agent_train.sft import native_config, run as run_sft, validate_dataset

ROOT = Path(__file__).resolve().parents[1]


def custom(tmp_path, values):
    path = tmp_path/'config.yml'
    path.write_text(yaml.safe_dump(values))
    return path


def test_profiles_inherit_defaults_and_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    expected = load_config()
    monkeypatch.chdir(tmp_path)
    for name in ('base', 'pilot', 'grpo_long', 'sft', 'coder_3b'):
        args = load_config(DEFAULT_CONFIG.with_name(name+'.yml'))
        assert args.model.is_absolute() and args.output.is_relative_to(ROOT/'data/training')
        assert args.rank == 8 and not args.train
    assert load_config().model == expected.model
    assert load_config(DEFAULT_CONFIG.with_name('grpo_long.yml')).steps == 200


def test_deep_override_cli_precedence_and_original_file_preserved(tmp_path):
    path = custom(tmp_path, {'train': {'type_cfg': {'learning_rate': '3e-6'}, 'lora': {'cfg': {'r': 16}}}})
    before = path.read_bytes()
    args = load_config(path, {'steps': 3, 'output': 'data/training/custom'})
    assert (args.steps, args.rank, args.learning_rate, args.group_size) == (3, 16, 3e-6, 4)
    assert args.output == ROOT/'data/training/custom'
    assert path.read_bytes() == before
    json.dumps(describe(args))


@pytest.mark.parametrize('values,match', [
    ({'train': {'qat': True}}, 'qat'),
    ({'train': {'type_cfg': {'use_vllm': True}}}, 'use_vllm'),
    ({'train': {'type_cfg': {'learning_rate': '.nan'}}}, 'learning_rate'),
    ({'train': {'type_cfg': {'num_generations': 1}}}, 'group_size'),
    ({'train': {'type_cfg': {'gradient_accumulation_steps': 8}}}, 'accumulation'),
    ({'train': {'type_cfg': {'typo_rate': 0.1}}}, 'Unknown'),
    ({'train': {'datasets': {'trajectory_paths': '...'}}}, 'trajectory_paths'),
    ({'train': {'datasets': {'os': ['']}}}, 'curriculum'),
    ({'train': {'lora': {'cfg': {'lora_dropout': .05}}}}, 'dropout'),
    ({'train': {'steps': True}}, 'boolean'),
    ({'runtime': {'gradient_checkpointing': 'false'}}, 'boolean'),
    ({'train': {'sft': {'mask_prompt': False}}}, 'mask_prompt'),
])
def test_unsupported_or_invalid_configuration_is_not_silently_ignored(tmp_path, values, match):
    with pytest.raises(ValueError, match=match): load_config(custom(tmp_path, values))


def test_inheritance_cycle_rejected(tmp_path):
    (tmp_path/'a.yml').write_text('extends: b.yml\n')
    (tmp_path/'b.yml').write_text('extends: a.yml\n')
    with pytest.raises(ValueError, match='cycle'): load_config(tmp_path/'a.yml')


def test_bad_yaml_explains_config_error(tmp_path):
    path = tmp_path/'invalid.yml'
    path.write_text('train: [unfinished')
    with pytest.raises(ValueError, match='Invalid YAML'): load_config(path)


def test_default_cli_never_imports_gpu_or_creates_outputs(tmp_path, capsys):
    output = tmp_path/'should-not-exist'
    assert main(['--output', str(output)]) == 0
    assert json.loads(capsys.readouterr().out)['train'] is False
    assert not output.exists()
    result = subprocess.run([sys.executable, '-c',
        'from src.agent_train.cli import main; import sys; main([]); assert "mlx.core" not in sys.modules'],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_sft_settings_reach_native_backend_and_empty_dataset_stops_before_launch(tmp_path, monkeypatch):
    args = load_config(DEFAULT_CONFIG.with_name('sft.yml'), {'output': str(tmp_path/'output')})
    args.dataset = tmp_path/'missing'
    native = native_config(args)
    assert native['mask_prompt'] and native['grad_accumulation_steps'] == 8
    assert native['max_seq_length'] == 2304 and native['lora_parameters']['rank'] == args.rank
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Training must not launch'))
    with pytest.raises(ValueError, match='absent or empty'): run_sft(args)
    assert not args.output.exists()


def test_sft_rejects_prompt_leakage_across_splits(tmp_path):
    row = {'messages': [{'role':'user','content':'task'}, {'role':'assistant','content':'answer'}]}
    (tmp_path/'train.jsonl').write_text(json.dumps(row)+'\n')
    row['messages'][-1]['content'] = 'another answer'
    (tmp_path/'valid.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='leakage'): validate_dataset(tmp_path)


def test_sft_timeout_leaves_reviewable_job_and_log(tmp_path, monkeypatch):
    args = load_config(DEFAULT_CONFIG.with_name('sft.yml'), {'output': str(tmp_path/'output')})
    args.dataset = tmp_path/'dataset'
    args.dataset.mkdir()
    row = {'messages': [{'role':'user','content':'task'}, {'role':'assistant','content':'answer'}]}
    (args.dataset/'train.jsonl').write_text(json.dumps(row)+'\n')
    def timeout(command, **kwargs):
        assert kwargs['timeout'] == args.max_seconds
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        job = json.loads(Path(command[-1]).read_text())
        assert job['dataset_rows'] == {'train': 1, 'valid': 0, 'test': 0}
        kwargs['stdout'].write('checkpoint saved\n')
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(ValueError, match='time limit'): run_sft(args)
    assert (args.output/'training.log').read_text() == 'checkpoint saved\n'


@pytest.mark.parametrize('row', [[], {'messages': [
    {'role':'user','content':'task'}, {'role':'assistant','content':'answer', 'tool_calls':[{}]}]}])
def test_sft_malformed_and_native_tool_rows_rejected(tmp_path, row):
    (tmp_path/'train.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError): validate_dataset(tmp_path)


def test_model_format_and_revision_checked_without_loading_weights(tmp_path):
    args = load_config(overrides={'model': str(tmp_path)})
    (tmp_path/'config.json').write_text('{"model_type":"qwen3","quantization":{"bits":4}}')
    with pytest.raises(ValueError, match='manifest'): validate_local_model(args)
    (tmp_path/'download_manifest.json').write_text(json.dumps({'repo_id':args.model_name,'revision':args.model_revision}))
    validate_local_model(args)
    (tmp_path/'config.json').write_text('{"model_type":"qwen3_vl","vision_config":{"layers":1}}')
    with pytest.raises(ValueError, match='Vision'): validate_local_model(args)


def test_old_imports_are_aliases_not_duplicate_implementations():
    from src.coding.trajectory import TrajectoryLog as old_log
    from src.agent_train.trajectory import TrajectoryLog
    from src.training.verified_edits import EditTask as old_task
    from src.agent_train.verified_edits import EditTask
    assert old_log is TrajectoryLog and old_task is EditTask
