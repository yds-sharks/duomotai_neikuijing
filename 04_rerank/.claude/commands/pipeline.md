# Pipeline 开发与执行

编写或执行多步骤处理任务时，严格遵守以下规范。

**操作记录要求：**
1. 开始前先读 `notes/ops/{任务名}_plan.md`，确认当前进度
2. 每步完成后更新 plan 进度标注，写 `notes/ops/{任务名}/stepNN_{步骤名}.md` 记录执行细节、中途调整和 bad cases
3. 执行完成后输出沉淀建议

---

## 一、强制三件套

### 1. 时间统计

- 脚本级总耗时
- 步骤级独立计时
- 长任务每N条输出进度、速度和ETA

```python
if completed % interval == 0:
    elapsed = time.time() - t0
    speed = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / speed if speed > 0 else 0
    print(f"  进度: {completed}/{total}, 耗时{elapsed:.0f}s, ETA {eta:.0f}s")
```

### 2. 流式落盘

长时间任务必须定期落盘，防止崩溃丢失进度。

```python
import threading

write_lock = threading.Lock()

def flush_to_disk():
    tmp_path = output_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, output_path)  # 原子操作

# 并发回调中：
with write_lock:
    results[sid] = result
    completed += 1
    if completed % flush_interval == 0:
        flush_to_disk()

# 循环结束后必须最终flush
flush_to_disk()
```

**踩坑经验（必须遵守）：**

1. **必须用 `os.replace()` 原子替换** — 直接写入如果中途崩溃，文件损坏且丢失所有之前结果
2. **并发必须加 `threading.Lock()`** — 不加锁导致：数据竞争、文件写入交错损坏、计数器不准
3. **flush_interval 策略** — 快速任务（<1s/条）每50条flush；慢速任务（>5s/条，如API调用）每5条flush。原则：最多丢失2-3分钟工作量
4. **循环结束后必须最终flush** — `as_completed` 结束时可能有未flush的尾部数据，漏掉 = 丢失最后一批
5. **日志和数据同步flush** — 两个文件在同一个flush函数中一起写，否则断点恢复时数据和日志不一致
6. **中间产物不用 `indent=2`** — indent格式化让5MB JSON膨胀到15MB+，flush慢3倍。中间产物用紧凑格式，最终输出才用indent

### 3. 断点续传

**模式A：基于输出文件恢复**
```python
done_sids = set()
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        existing = json.load(f)
    done_sids = set(existing.keys())
    results.update(existing)  # 关键：加载已有结果到内存
    print(f"  断点恢复: 已完成{len(done_sids)}条")

todo_ids = [sid for sid in all_ids if sid not in done_sids]
```

**模式B：基于CheckpointManager**
```python
ckpt = CheckpointManager(output_dir, 'stage_name', flush_interval=5)
done_ids = ckpt.completed_ids()
todo = [(i, task) for i, task in enumerate(tasks) if f"key_{i}" not in done_ids]
# ... 处理 ...
ckpt.flush_final()
```

**踩坑经验：**
1. 恢复时**必须加载已有数据到内存** — 否则flush时覆盖掉之前的结果
2. 断点恢复通过 `--resume` 参数控制 — 不要默认自动恢复
3. checkpoint文件用 `_ckpt_` 前缀 — 和正式输出区分，任务完成后清理

---

## 二、输入数据校验

脚本启动时先校验输入，在处理前暴露问题：

1. **字段完整性**: 采样检查必需字段是否存在
2. **有效值**: 统计关键字段为空的比例
3. **API探测**: API类任务先用1条数据试调用
4. 有ERROR则停止，有WARN则报告用户

---

## 三、基础测试

代码开发完成后必须：

1. **抽样测试**: 用 `--sample_ids` 或 `--max_samples 3` 跑通全流程
2. **边界case**: 空列表、None值、超长字段不能报错
3. **输出校验**: 输出文件可正常 json.load，样本数符合预期

---

## 四、沉淀提醒

每次Pipeline或任务完成后，**必须**输出：

```
=== 沉淀提醒 ===
本次产出（tmp/中）:
  - {文件名} ({大小}) — {说明}

建议沉淀到 persistent/:
  - {文件} — 原因：{最终产出/关键中间结果}

建议记录的 bad case:
  - {样本ID}: {问题描述}

等待你的指令。
```

不主动执行任何移动或删除操作。
