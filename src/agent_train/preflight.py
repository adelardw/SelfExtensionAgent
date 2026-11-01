"""Filesystem-only checks before loading the optional GPU backend."""
import json


def validate_local_model(args):
    if not args.model.is_dir(): raise ValueError('model.path must be a downloaded local MLX directory; no automatic download')
    path = args.model / 'config.json'
    config = json.loads(path.read_text())
    if config.get('vision_config') or 'vl' in config.get('model_type', '').lower():
        raise ValueError('Vision-language training needs a separate backend; this runner handles text models only')
    if not isinstance(config.get('quantization'), dict):
        raise ValueError('QLoRA requires a quantized MLX model, not GGUF or unquantized HF weights')
    if args.model_revision:
        manifest = args.model / 'download_manifest.json'
        info = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if info.get('revision') != args.model_revision or info.get('repo_id') != args.model_name:
            raise ValueError('Local download_manifest does not match configured model.name/revision')
    if args.init_adapter is not None:
        for name in ('adapter_config.json', 'adapters.safetensors', 'sft-job.json'):
            if not (args.init_adapter/name).is_file():
                raise ValueError('Initial adapter requires completed SFT artifacts: ' + name)
        settings = json.loads((args.init_adapter/'sft-job.json').read_text())['settings']
        if settings.get('model_name') != args.model_name or settings.get('model_revision') != args.model_revision:
            raise ValueError('Initial SFT adapter belongs to a different base model/revision')
