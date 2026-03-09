# handler.py - FireRed-Image-Edit RunPod Serverless Handler
import base64
import io
import os
import torch
import runpod
from PIL import Image

# 优先用 Network Volume 里的模型，否则从 HuggingFace 下载
_VOLUME_PATH = "/runpod-volume/models/FireRed-Image-Edit-1.1"
_HF_MODEL_ID = os.environ.get("MODEL_ID", "FireRedTeam/FireRed-Image-Edit-1.1")
MODEL_ID = _VOLUME_PATH if os.path.isdir(_VOLUME_PATH) else _HF_MODEL_ID

# 把 transformers 里存在但未导出的类注入到顶层命名空间
# 解决 diffusers 用 `from transformers import Qwen2_5_VLForConditionalGeneration` 找不到的问题
import transformers
print(f"[PATCH] transformers version: {transformers.__version__}")
_patches = {
    "Qwen2_5_VLForConditionalGeneration": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    "Qwen2VLProcessor": "transformers.models.qwen2_vl.processing_qwen2_vl",
    "Qwen2Tokenizer": "transformers.models.qwen2.tokenization_qwen2",
}
import importlib as _il
for _cls_name, _module_path in _patches.items():
    # 用 __dict__ 检查，避免触发 transformers 的 lazy import __getattr__
    if _cls_name not in transformers.__dict__:
        try:
            _mod = _il.import_module(_module_path)
            _cls = getattr(_mod, _cls_name)
            setattr(transformers, _cls_name, _cls)
            print(f"[PATCH] injected {_cls_name}")
        except Exception as _e:
            print(f"[PATCH] failed to inject {_cls_name}: {_e}")
    else:
        print(f"[PATCH] {_cls_name} already in transformers")

# 全局 pipeline 缓存（懒加载）
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from diffusers import QwenImageEditPlusPipeline

    print(f"Loading pipeline: {MODEL_ID}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    _pipeline = pipe
    print("Pipeline loaded.")
    return _pipeline


def handler(job):
    try:
        inp = job["input"]
        prompt = inp["prompt"]
        image_b64 = inp["image_b64"]

        # 可选参数
        seed = inp.get("seed", 49)
        true_cfg_scale = inp.get("true_cfg_scale", 4.0)
        num_inference_steps = inp.get("num_inference_steps", 40)
        negative_prompt = inp.get("negative_prompt", " ")

        # 解码输入图片
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        pipeline = get_pipeline()

        inputs = {
            "image": [image],
            "prompt": prompt,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": 1,
        }

        with torch.inference_mode():
            result = pipeline(**inputs)

        output_image = result.images[0]

        # 编码输出图片为 base64
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        output_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "mime": "image/png",
            "image_b64": output_b64,
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
