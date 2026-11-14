# Krea2 + Anima 双阶段生成

一个用于 Forge Neo 文生图页面的双模型脚本。两种顺序都可选择：第一阶段文生图，第二阶段以第一阶段的图进行图生图。

- **Anima → Krea2**：先由 Anima 生成角色，再交给 Krea2 重绘，适合以 Krea2 为主要出图模型。
- **Krea2 → Anima**：先由 Krea2 生成，再交给 Anima 重绘。
- 固定 LoRA 按模型分组；可选保留第一阶段图片；支持在第二阶段后使用主面板 Hires.fix。

## 安装

1. 下载 [krea2_anima.py](scripts/krea2_anima.py)，放入 Forge 根目录的 `scripts` 文件夹。
2. 将模型、文本编码器和 VAE 放入 Forge 对应的模型目录，刷新模型列表或重启 Forge。
3. 在 **txt2img / 文生图**页面的 Script 下拉菜单中选择 **Krea2 + Anima 双阶段生成**。

此脚本针对 Forge Neo 的模型加载和处理接口编写；其他版本的 Forge 可能需要适配。仓库不包含模型权重。

## 模型选择

| 用途 | 本脚本的默认选择关键字 | 组件 |
| --- | --- | --- |
| Krea2 | `krea2_turbo_fp8_scaled` | Krea2 Turbo FP8 checkpoint |
| Krea2 编码器 | `qwen3vl_4b_fp8_scaled` | Qwen3-VL 4B |
| Anima | `waiANIMA_v10Base10` | Anima Base checkpoint |
| Anima 编码器 | `qwen_3_06b_base` | Qwen3 0.6B |
| 共享 VAE | `qwen_image_vae` | Qwen Image VAE |

请按实际安装的文件选择各自的模型和编码器，两个阶段共享所选 VAE。脚本在切换模型时一并切换编码器，第一阶段的图像经解码后交给第二阶段重新编码。模型文件名不完全相同时，可手动选择；确保模型架构和组件配套。

## 使用

1. 在主 Prompt 填写提示词，选择阶段顺序。如果希望 Anima 先识别角色、Krea2 负责最终画面，选择 **Anima → Krea2**。
2. 选择两个模型、各自的编码器和共享 VAE。「第一阶段固定提示词」留空时，两阶段逐张使用主提示词或注入后的缓存提示词；填写后只覆盖第一阶段的提示词。
3. 设置主面板的尺寸、种子、批次数量，以及脚本内的采样参数。
4. 「第二阶段重绘幅度」决定第二阶段对输入图的改动程度，默认 `0.60`。增大通常会带来更多重绘，也可能改变角色或姿势；按效果调整。
5. 勾选「保留第一阶段图」后，第一阶段图片会显示并保存到样本目录下对应的 `anima-stage-1` 或 `krea2-stage-1` 目录；未勾选时只用于后续生成。Forge 禁止保存的设置仍然生效。

| 默认参数 | Krea2 Turbo | Anima Base |
| --- | --- | --- |
| 步数 | 8 | 28 |
| 采样器 | Euler | Euler a |
| 调度器 | Simple | Normal |
| CFG | 固定 1 | 4.5 |
| Shift | 模型默认 | 3 |


## 固定 LoRA

将完整标签写在 Forge 原始主 Prompt 中，例如：

```text
1girl, blue eyes, <lora:krea_style:0.7>, <lora:anime_character:0.9>
```

然后在脚本顶部分别填写：

- **Krea2 LoRA 标签**：`<lora:krea_style:0.7>`
- **Anima LoRA 标签**：`<lora:anime_character:0.9>`

各框可留空；多个标签可用逗号或换行分隔。每个主 Prompt 中的固定 LoRA 必须分配给且只分配给一个模型，完整标签中的名称和权重需与主 Prompt 一致。LoRA 跟随所属模型，不随阶段顺序改变；选用与对应模型兼容的 LoRA。

## Hires.fix

在主面板开启 Hires.fix 后，脚本在第二阶段成图上继续执行高分修复。高分阶段使用第二模型及其编码器、VAE 和 LoRA，放大器、倍率或目标尺寸、步数、采样器、调度器和重绘幅度等来自主面板。Krea2 的 CFG 固定为 1；第二模型是 Anima 时使用主面板的高分 CFG。高分提示词留空时继承对应图片的提示词。

脚本的「第二阶段重绘幅度」和主面板的高分重绘幅度分别控制两个环节。主面板保存放大前图片的选项与「保留第一阶段图」分别控制不同阶段。

## LLM Prompt Studio 和无限生成

该脚本提供缓存桥接入口，可与已支持此双阶段脚本的 [LLM Prompt Studio](https://github.com/Rivulet138/sd-webui-llm-prompt-studio) 配合：在文生图页面启用缓存 Prompt 注入并选择缓存库，然后使用 Forge 主生成按钮的原生无限生成。每轮按生成数量领取缓存，两个阶段使用每张图片对应的提示词；固定 LoRA 仍在原始 Prompt 中填写并按模型分组。

## 验证范围

本地自动化测试覆盖阶段顺序、LoRA 分流及缓存桥接；最新缓存桥接与 Hires.fix 路径尚未完成完整 GPU 端到端验证。实际效果取决于模型、组件及参数。
