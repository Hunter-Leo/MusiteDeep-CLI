# 需求 001：CLI 支持自定义模型路径

## Project Stage

```
project_stage: pre-launch
```

## Spec

### 背景

目前 `musitedeep` CLI 的模型路径硬编码在 `cli.py` 中：

- `MUSITEDEEP_DIR` 固定为脚本所在目录
- `check_model_data()` 只检查 `<project_dir>/models/` 下的模型
- `run_single_model_prediction()` 的 `-model-prefix` 参数固定为 `models/{model_name}`

用户无法指定自定义模型路径，当模型文件存放在其他位置时只能通过符号链接或复制到 `models/` 目录来绕开。

### 目标

为 `musitedeep` CLI 添加 `--model-dir` / `-M` 选项，允许用户通过命令行指定模型文件所在的目录路径。

### 使用场景

```bash
# 默认行为不变（从项目目录下的 models/ 加载）
musitedeep -s "MKTVRQERLK" -m py

# 指定自定义模型路径
musitedeep -s "MKTVRQERLK" -m py --model-dir /data/models/musitedeep
musitedeep -s "MKTVRQERLK" -m all -M /data/models/musitedeep
```

## Requirements

### 功能需求

1. **新增 `--model-dir` / `-M` 选项**：接受一个目录路径作为模型文件根目录
2. **向后兼容**：不指定 `--model-dir` 时，行为与当前完全一致（默认为 `MUSITEDEEP_DIR/models/`）
3. **路径传递**：自定义路径需正确传递给 `check_model_data()` 和 `predict_multi_batch.py` 的 `-model-prefix` 参数
4. **路径验证**：如果指定的路径不存在，给出明确的错误提示

### 技术需求

- 修改文件：仅 `cli.py`
- 调用链：`predict()` → `check_model_data(models_dir)` → `run_single_model_prediction(..., model_prefix)`
- `-model-prefix` 传递给 `predict_multi_batch.py` 时，应是相对于项目目录或绝对路径

## Action Items

```
**Prerequisite documents** (根据 LLM 判断是否需要):
- [ ] `generated/inspect.md` — 现有代码分析（Phase 02）

**Round artifacts** (跨轮次维护):
- [ ] `issues.md` — 轮次间的 Issue 日志

**Required documents** (始终需要，按顺序):
- [ ] `generated/rounds/round-001/plan.md` — Phase 04（实施计划）
- [ ] `generated/rounds/round-001/tasks.md` — Phase 05（任务规划）
- [ ] `generated/start-and-resume.md` — Phase 06（启动与恢复指南）
- [ ] `cli.py` 代码修改 — Phase 07（执行）
```

## Constitution

### 编程规范

- 所有标识符、注释和文档使用英文（CLI 帮助文本保持英文）
- Python snake_case 命名
- 函数参数添加类型注解
- 错误信息使用 Click 的 `click.echo(..., err=True)` 输出到 stderr
- 保持与现有 CLI 代码风格一致

### 兼容性

- `--model-dir` 默认值与当前硬编码路径一致，完全向后兼容
- 不改变现有命令行接口的语义
