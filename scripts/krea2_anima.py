"""Sequential Krea 2 and Anima workflow for Forge Neo."""

import copy
import os
import re

import gradio as gr

from modules import processing, scripts, sd_models, sd_samplers, sd_schedulers, shared
from modules.sysinfo import set_config
from modules_forge import main_entry


TITLE = "Krea2 + Anima 双阶段生成"
STAGE_ORDERS = ("Krea2 → Anima", "Anima → Krea2")
RESTORED_OPTIONS = (
    "sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype",
    "krea2_do_reference", "anima_do_reference", "img2img_fix_steps",
    "img2img_color_correction", "sd_vae_encode_method", "sd_vae_decode_method",
)
LORA_TAG_RE = re.compile(r"<lora:[^<>]+>", re.IGNORECASE)


def _extract_lora_tags(prompt):
    """Return fixed Forge LoRA tags from the original parent prompt."""
    return LORA_TAG_RE.findall(str(prompt or ""))


def _lora_tag_name(tag):
    body = tag[len("<lora:"):-1]
    if ":" not in body:
        raise ValueError(f"LoRA 标签格式无效：{tag}。请使用 <lora:名称:权重>。")
    return body.split(":", 1)[0].strip()


def _parse_lora_assignments(value, model_name):
    entries = [entry.strip() for entry in re.split(r"[,，;；\n]+", str(value or "")) if entry.strip()]
    assignments = []
    for entry in entries:
        if entry.casefold().startswith("<lora:"):
            if not entry.endswith(">"):
                raise ValueError(f"{model_name} LoRA 标签格式无效：{entry}。请使用 <lora:名称:权重>。")
            assignments.append((_lora_tag_name(entry), entry))
        else:
            assignments.append((entry, None))

    names = [name for name, _ in assignments]
    normalized = [name.casefold() for name in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{model_name} LoRA 标签列表中有重复项。")
    return assignments


def _group_fixed_lora_tags(tags, krea_names, anima_names):
    """Match parent-prompt LoRAs to their compatible model by Forge tag name."""
    names_by_model = {
        "Krea2": _parse_lora_assignments(krea_names, "Krea2"),
        "Anima": _parse_lora_assignments(anima_names, "Anima"),
    }
    assignments = {}
    for model_name, entries in names_by_model.items():
        for name, configured_tag in entries:
            normalized = name.casefold()
            if normalized in assignments:
                raise ValueError(f"LoRA「{name}」不能同時歸入 Krea2 和 Anima。")
            assignments[normalized] = (model_name, configured_tag)

    grouped = {"Krea2": [], "Anima": []}
    prompt_names = set()
    unassigned = []
    mismatched = []
    for tag in tags:
        name = _lora_tag_name(tag)
        normalized = name.casefold()
        prompt_names.add(normalized)
        assignment = assignments.get(normalized)
        if assignment is None:
            unassigned.append(tag)
        else:
            model_name, configured_tag = assignment
            if configured_tag and configured_tag.casefold() != tag.casefold():
                mismatched.append((configured_tag, tag))
            grouped[model_name].append(tag)

    unused = [name for name in assignments if name not in prompt_names]
    if unassigned or unused or mismatched:
        details = []
        if unassigned:
            details.append("未归组标签：" + ", ".join(unassigned))
        if unused:
            details.append("原始 Prompt 中未找到名称：" + ", ".join(
                name for entries in names_by_model.values() for name, _ in entries
                if name.casefold() in unused
            ))
        if mismatched:
            details.append("完整标签与原始 Prompt 不一致：" + ", ".join(
                f"{configured}（Prompt 中为 {actual}）" for configured, actual in mismatched
            ))
        raise ValueError("固定 LoRA 配置无效。" + "；".join(details) +
                         "。请检查两个模型的 LoRA 标签列表。")
    return grouped


def _strip_lora_tags(prompt):
    """Remove loader-only LoRA tags before sending a prompt to Krea2."""
    text = LORA_TAG_RE.sub("", str(prompt or ""))
    return re.sub(r"\s*[,，]\s*(?=[,，])", ", ", text).strip(" ,，")


def _ensure_fixed_lora(prompt, fixed_lora_tags):
    """Keep every fixed parent LoRA exactly once in an Anima prompt."""
    result = str(prompt or "").strip()
    existing = {tag.casefold() for tag in LORA_TAG_RE.findall(result)}
    missing = [tag for tag in fixed_lora_tags if tag.casefold() not in existing]
    if missing:
        result = ", ".join([result, *missing]) if result else ", ".join(missing)
    return result


def _route_lora(prompt, stage_lora_tags):
    """Keep only fixed parent-prompt LoRAs assigned to this model stage."""
    return _ensure_fixed_lora(_strip_lora_tags(prompt), stage_lora_tags)


def _default(choices, filename):
    return next((x for x in choices if filename in x), None)


def _cancelled():
    state = shared.state
    return state.interrupted or state.stopping_generation or state.skipped


def _configure(values):
    set_config(values, is_api=True, run_callbacks=False, save_config=False)
    # set_config does not refresh loading parameters for a dtype-only change.
    main_entry.refresh_model_loading_parameters()


def _resolve_checkpoint(value):
    # Do not use fuzzy matching here: a typo must not load an unrelated model.
    info = sd_models.checkpoint_aliases.get(value)
    if info is None:
        raise ValueError(f"找不到模型：{value}。请点击刷新模型并重新选择。")
    return info.title


def _resolve_module(value):
    name = os.path.basename(value or "")
    path = main_entry.module_list.get(name)
    if path is None or not os.path.isfile(path):
        raise ValueError(f"找不到编码器 / VAE：{value}。请检查 models/text_encoder 与 models/VAE。")
    return str(path)


def _entries(result):
    """Exclude grids, keeping gallery entries and PNG metadata in sync."""
    first = result.index_of_first_image
    return [
        (im, result.infotexts[first + i], result.all_prompts[i],
         result.all_negative_prompts[i], result.all_seeds[i], result.all_subseeds[i])
        for i, im in enumerate(result.images[first:])
    ]


def _aggregate(result, entries):
    result.images = [x[0] for x in entries]
    result.extra_images = []
    result.index_of_first_image = 0
    if entries:
        result.infotexts = [x[1] for x in entries]
        result.all_prompts = [x[2] for x in entries]
        result.all_negative_prompts = [x[3] for x in entries]
        result.all_seeds = [x[4] for x in entries]
        result.all_subseeds = [x[5] for x in entries]
        result.info, result.prompt, result.negative_prompt, result.seed, result.subseed = entries[0][1:]
    return result


def _native_cache_prompts(p, total):
    """Consume Studio's parent batch once; child stages run without hooks."""
    for bridge in getattr(getattr(p, "scripts", None), "alwayson_scripts", ()):
        if bridge.title() != "LLM Prompt Studio native cache injection":
            continue
        apply_cache = getattr(bridge, "apply_native_cache_for_script", None)
        if not callable(apply_cache):
            raise ValueError("请更新 LLM Prompt Studio 缓存桥接并重启 Forge。")
        previous_styles = getattr(p, "styles", None)
        p.styles = []
        try:
            token = apply_cache(p)
        finally:
            p.styles = previous_styles
        if token:
            prompts = list(p.all_prompts)
            if len(prompts) != total or not all(isinstance(prompt, str) for prompt in prompts):
                raise ValueError("缓存提示词数量与本次生成数量不一致。")
            return prompts
        break
    return [p.prompt] * total


class Script(scripts.Script):
    supports_native_cache_injection = True

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        models, modules = main_entry.refresh_models()
        samplers = sd_samplers.visible_sampler_names()
        schedules = [s.label for s in sd_schedulers.schedulers]
        with gr.Row():
            krea_lora_names = gr.Textbox(label="Krea2 LoRA 标签", placeholder="例如 <lora:krea_style:0.7>")
            anima_lora_names = gr.Textbox(label="Anima LoRA 标签", placeholder="例如 <lora:anime_character:0.9>")
        with gr.Row():
            krea = gr.Dropdown(models, value=_default(models, "krea2_turbo_fp8_scaled"), label="Krea2 模型")
            anima = gr.Dropdown(models, value=_default(models, "waiANIMA_v10Base10"), label="Anima 模型")
        with gr.Row():
            stage_order = gr.Dropdown(list(STAGE_ORDERS), value=STAGE_ORDERS[0], label="阶段顺序")
        with gr.Row():
            kenc = gr.Dropdown(modules, value=_default(modules, "qwen3vl_4b_fp8_scaled"), label="Krea2 编码器")
            aenc = gr.Dropdown(modules, value=_default(modules, "qwen_3_06b_base"), label="Anima 编码器")
            vae = gr.Dropdown(modules, value=_default(modules, "qwen_image_vae"), label="共享 VAE")
        refresh = gr.Button("刷新模型和编码器", size="sm")

        def refresh_choices(*values):
            new_models, new_modules = main_entry.refresh_models()
            choices = [new_models, new_models, new_modules, new_modules, new_modules]
            return [gr.update(choices=c, value=v if v in c else None) for c, v in zip(choices, values)]

        selections = [krea, anima, kenc, aenc, vae]
        refresh.click(refresh_choices, inputs=selections, outputs=selections, show_progress=False)
        composition = gr.Textbox(label="第一阶段固定提示词（可选）", lines=3, placeholder="留空时使用每张图注入后的主提示词；填写后第一阶段固定使用此文本")
        with gr.Row():
            ksteps = gr.Slider(1, 50, value=8, step=1, label="Krea2 步数（CFG 固定 1）")
            ksampler = gr.Dropdown(samplers, value="Euler", label="Krea2 采样器")
            kschedule = gr.Dropdown(schedules, value="Simple", label="Krea2 调度器")
        with gr.Row():
            asteps = gr.Slider(1, 100, value=28, step=1, label="Anima 实际采样步数")
            acfg = gr.Slider(1, 12, value=4.5, step=0.1, label="Anima CFG")
            asampler = gr.Dropdown(samplers, value="Euler a", label="Anima 采样器")
            aschedule = gr.Dropdown(schedules, value="Normal", label="Anima 调度器")
        with gr.Row():
            shift = gr.Slider(0.1, 10, value=3, step=0.05, label="Anima Shift")
            denoise = gr.Slider(0.05, 1, value=0.60, step=0.01, label="第二阶段重绘幅度（越高改动越大）")
        with gr.Row():
            save = gr.Checkbox(value=False, label="保留第一阶段图")
            # Reserve the old positional argument so existing API requests remain aligned.
            show = gr.Checkbox(value=False, visible=False)
        return [*selections, composition, ksteps, ksampler, kschedule,
                asteps, acfg, asampler, aschedule, shift, denoise, save, show,
                stage_order, krea_lora_names, anima_lora_names]

    def run(self, p, krea_model, anima_model, krea_encoder, anima_encoder, vae,
            composition_prompt, krea_steps, krea_sampler, krea_schedule,
            anima_steps, anima_cfg, anima_sampler, anima_schedule, anima_shift,
            denoise, save_drafts, show_drafts, stage_order=STAGE_ORDERS[0],
            krea_lora_names="", anima_lora_names=""):
        main_entry.refresh_models()
        krea_model = _resolve_checkpoint(krea_model)
        anima_model = _resolve_checkpoint(anima_model)
        krea_encoder = _resolve_module(krea_encoder)
        anima_encoder = _resolve_module(anima_encoder)
        vae = _resolve_module(vae)
        if krea_model == anima_model or krea_encoder == anima_encoder:
            raise ValueError("Krea2 与 Anima 必须选择各自的模型和不同的文本编码器。")
        if stage_order not in STAGE_ORDERS:
            raise ValueError(f"未知阶段顺序：{stage_order}")
        for sampler, schedule in [(krea_sampler, krea_schedule), (anima_sampler, anima_schedule)]:
            if sampler not in sd_samplers.all_samplers_map or schedule not in sd_schedulers.schedulers_map:
                raise ValueError(f"未知采样器或调度器：{sampler} / {schedule}")
        if not 0 < float(denoise) <= 1:
            raise ValueError("第二阶段重绘幅度必须大于 0 且不超过 1。")
        if not 1 <= int(krea_steps) <= 50 or not 1 <= int(anima_steps) <= 100:
            raise ValueError("采样步数超出范围。")
        if not 1 <= float(anima_cfg) <= 12 or not 0.1 <= float(anima_shift) <= 10:
            raise ValueError("Anima CFG 或 Shift 超出范围。")
        if not isinstance(p.prompt, str) or not isinstance(p.negative_prompt, str):
            raise ValueError("本流程使用一组主提示词；批量请通过主面板数量设置。")

        fixed_lora_tags = _extract_lora_tags(p.prompt)
        lora_tags_by_model = _group_fixed_lora_tags(
            fixed_lora_tags, krea_lora_names, anima_lora_names,
        )
        processing.fix_seed(p)
        total = int(p.batch_size) * int(p.n_iter)
        if total < 1:
            raise ValueError("生成数量至少为 1。")
        # Studio fills an inherited hr_prompt with its first cached prompt during injection.
        # Capture user overrides first so each hires image can inherit its own cache entry.
        hires_prompt = getattr(p, "hr_prompt", "")
        hires_negative_prompt = getattr(p, "hr_negative_prompt", "")
        prompts = _native_cache_prompts(p, total)
        composition = composition_prompt.strip()
        first_model, second_model = stage_order.split(" → ")
        model_settings = {
            "Krea2": {
                "checkpoint": krea_model, "encoder": krea_encoder,
                "steps": int(krea_steps), "cfg": 1.0,
                "sampler": krea_sampler, "schedule": krea_schedule,
            },
            "Anima": {
                "checkpoint": anima_model, "encoder": anima_encoder,
                "steps": int(anima_steps), "cfg": float(anima_cfg),
                "sampler": anima_sampler, "schedule": anima_schedule,
            },
        }
        snapshot = {k: copy.deepcopy(getattr(shared.opts, k)) for k in RESTORED_OPTIONS}
        drafts, finals = [], []
        last_result = None
        enable_hr = bool(getattr(p, "enable_hr", False))
        progress_refined = shared.state.processing_has_refined_job_count
        # Native standalone hires advances nextjob twice; each other stage advances once.
        shared.state.job_count = total * (4 if enable_hr else 2)

        def make_stage(kind, seed, model_name, prompt, negative_prompt, **params):
            stage_lora_tags = lora_tags_by_model[model_name]
            stage = kind(
                prompt=_route_lora(prompt, stage_lora_tags),
                negative_prompt=negative_prompt, styles=[], seed=seed,
                subseed=seed, subseed_strength=0, seed_enable_extras=False,
                batch_size=1, n_iter=1, width=p.width, height=p.height,
                outpath_samples=os.path.join(p.outpath_samples, f"{model_name.lower()}-stage-1")
                if kind is processing.StableDiffusionProcessingTxt2Img else p.outpath_samples,
                outpath_grids=p.outpath_grids,
                do_not_save_samples=p.do_not_save_samples, do_not_save_grid=True,
                restore_faces=False, tiling=False,
                disable_extra_networks=not stage_lora_tags,
                refiner_checkpoint=None, override_settings={}, extra_generation_params={},
            )
            stage.user = p.user
            stage.is_api = p.is_api
            for key, value in params.items():
                setattr(stage, key, value)
            return stage

        def generate(stage):
            try:
                return processing.process_images(stage)
            finally:
                stage.close()

        try:
            if enable_hr:
                # Own the whole workflow total instead of letting each hires child double it.
                shared.state.processing_has_refined_job_count = True
                shared.total_tqdm.updateTotal(total * (
                    int(krea_steps) + int(anima_steps)
                    + (getattr(p, "hr_second_pass_steps", 0) or model_settings[second_model]["steps"])
                ))
            if not _cancelled():
                _configure({
                    "sd_model_checkpoint": model_settings[first_model]["checkpoint"],
                    "forge_additional_modules": [model_settings[first_model]["encoder"], vae],
                    "forge_unet_storage_dtype": "Automatic",
                    "krea2_do_reference": False, "anima_do_reference": False,
                    "img2img_fix_steps": True, "img2img_color_correction": False,
                    "sd_vae_encode_method": "Full", "sd_vae_decode_method": "Full",
                })
            for i in range(total):
                if _cancelled():
                    break
                shared.state.job = f"{first_model} 第一阶段 {i + 1}/{total}"
                first_settings = model_settings[first_model]
                first_prompt = composition or prompts[i]
                first_negative = "" if first_model == "Krea2" else p.negative_prompt
                first_stage_label = "Krea2 composition" if first_model == "Krea2" else "Anima draft"
                first_params = {
                    "steps": first_settings["steps"], "cfg_scale": first_settings["cfg"],
                    "sampler_name": first_settings["sampler"], "scheduler": first_settings["schedule"],
                    "outpath_samples": os.path.join(p.outpath_samples, f"{first_model.lower()}-stage-1"),
                    "do_not_save_samples": p.do_not_save_samples or not save_drafts,
                    "extra_generation_params": {
                        "Workflow": stage_order, "Workflow stage": first_stage_label,
                        f"{first_model} prompt": first_prompt,
                    },
                }
                if first_model == "Anima":
                    first_params["distilled_cfg_scale"] = float(anima_shift)
                stage = make_stage(
                    processing.StableDiffusionProcessingTxt2Img,
                    (int(p.seed) + i) % 4294967294, first_model, first_prompt, first_negative,
                    **first_params,
                )
                result = generate(stage)
                # Interrupted sampling may return a partially denoised image: do not feed it to stage 2.
                if _cancelled():
                    break
                drafts.extend(_entries(result))
                last_result = result

            if drafts and not _cancelled():
                _configure({
                    "sd_model_checkpoint": model_settings[second_model]["checkpoint"],
                    "forge_additional_modules": [model_settings[second_model]["encoder"], vae],
                })
                for i, draft in enumerate(drafts):
                    if _cancelled():
                        break
                    shared.state.job = f"{second_model} 第二阶段 {i + 1}/{len(drafts)}"
                    second_settings = model_settings[second_model]
                    second_negative = "" if second_model == "Krea2" else p.negative_prompt
                    second_prompt = prompts[i]
                    second_stage_label = "Anima final" if second_model == "Anima" else "Krea2 final"
                    second_params = {
                        "init_images": [draft[0].convert("RGB")], "resize_mode": 0,
                        "steps": second_settings["steps"], "cfg_scale": second_settings["cfg"],
                        "sampler_name": second_settings["sampler"], "scheduler": second_settings["schedule"],
                        "denoising_strength": float(denoise),
                        "do_not_save_samples": p.do_not_save_samples or enable_hr,
                        "extra_generation_params": {
                            "Workflow": stage_order, "Workflow stage": second_stage_label,
                            "Stage 1 model": first_model, "Stage 1 prompt": draft[2],
                            "Stage 1 seed": draft[4],
                            "Krea2 model": krea_model, "Anima model": anima_model,
                            "Krea2 encoder": os.path.basename(krea_encoder),
                            "Anima encoder": os.path.basename(anima_encoder),
                            "Workflow VAE": os.path.basename(vae),
                        },
                    }
                    if second_model == "Anima":
                        second_params["distilled_cfg_scale"] = float(anima_shift)
                    if first_model == "Krea2":
                        second_params["extra_generation_params"].update({
                            "Krea2 prompt": draft[2], "Krea2 seed": draft[4],
                            "Krea2 steps": int(krea_steps), "Krea2 CFG": 1,
                            "Krea2 sampler": krea_sampler, "Krea2 schedule": krea_schedule,
                        })
                    stage = make_stage(
                        processing.StableDiffusionProcessingImg2Img, draft[4], second_model,
                        second_prompt, second_negative, **second_params,
                    )
                    result = generate(stage)
                    if _cancelled():
                        break
                    last_result = result
                    for entry in _entries(result):
                        # Keep a completed second-stage image if hires is later interrupted.
                        finals.append(entry)
                        if not enable_hr or _cancelled():
                            continue
                        shared.state.job = f"{second_model} 高分辨率修复 {i + 1}/{len(drafts)}"
                        hr_params = {
                            key: getattr(p, key) for key in (
                                "hr_scale", "hr_upscaler", "hr_second_pass_steps", "hr_resize_x", "hr_resize_y",
                                "hr_sampler_name", "hr_scheduler", "denoising_strength",
                            ) if hasattr(p, key)
                        }
                        hr_params.update({
                            "enable_hr": True, "txt2img_upscale": True,
                            "firstpass_image": entry[0].convert("RGB"),
                            "width": entry[0].width, "height": entry[0].height,
                            "outpath_samples": p.outpath_samples,
                            "steps": second_settings["steps"], "cfg_scale": second_settings["cfg"],
                            "sampler_name": second_settings["sampler"], "scheduler": second_settings["schedule"],
                            "hr_checkpoint_name": None, "hr_additional_modules": None,
                            "hr_prompt": _route_lora(
                                hires_prompt or entry[2], lora_tags_by_model[second_model],
                            ),
                            "hr_negative_prompt": "" if second_model == "Krea2" else (
                                hires_negative_prompt or entry[3]
                            ),
                            "hr_cfg": 1.0 if second_model == "Krea2" else getattr(p, "hr_cfg", float(anima_cfg)),
                            "hr_distilled_cfg": getattr(p, "hr_distilled_cfg", float(anima_shift)),
                            "extra_generation_params": {
                                **second_params["extra_generation_params"],
                                "Workflow stage": f"{second_model} hires",
                                "Stage 2 denoising strength": float(denoise),
                            },
                        })
                        if second_model == "Anima":
                            hr_params["distilled_cfg_scale"] = float(anima_shift)
                        hr_stage = make_stage(
                            processing.StableDiffusionProcessingTxt2Img, entry[4], second_model,
                            entry[2], entry[3], **hr_params,
                        )
                        hr_result = generate(hr_stage)
                        if _cancelled():
                            break
                        hr_entries = _entries(hr_result)
                        if hr_entries:
                            finals[-1:] = hr_entries
                            last_result = hr_result
            if last_result is None:
                last_result = processing.Processed(p, [], seed=p.seed)
            # The visible retain switch controls both disk and gallery; show_drafts is legacy-only.
            return _aggregate(last_result, finals + (drafts if save_drafts else []))
        finally:
            try:
                _configure(snapshot)
            finally:
                shared.state.processing_has_refined_job_count = progress_refined
