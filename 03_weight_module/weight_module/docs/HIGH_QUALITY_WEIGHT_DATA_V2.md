# 高质量权重模块数据重构方案 V2

## 1. 这次重构要解决什么
- 旧训练数据主要围绕 `final_description -> 模板化 query 链` 构造，离真实测试 query 太远。
- 旧方案把 `density_level` 近似做成了“槽位个数/句长等级”，这不符合真实目标。
- 真实目标应该是：文本里有没有“小范围、可检索的限定锚点”。
- 真实测试集的大头不是医生自由检索问句，而是 benchmark 题干式 query。
- 旧锚点中大量文本其实是图说、流程、教学说明、多图联动描述，对检索监督价值很弱。
- `R1-R3` 不再表示“图片依赖程度评级”，而表示“通道倾向”，并且不能从 `density_level` 派生。

## 2. 真实测试 query 长什么样
基于：
- `/mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/translate_benchmark/output/endobench_test_translated.jsonl`
- `/mnt/data_1/yds/多模态/responses_run1.jsonl`

结论：
- 总量 `6832`
- 去重后 `2582`
- 长度分布：`min=13 median=27 p90=43 max=84`
- 主任务集中在：
  - `Region Recognition`
  - `Lesion Type Identification`
  - `Macro Phases Identification`
  - `Visual Grounding`
  - `Landmark Identification`
  - `Organ Identification`
- 主场景集中在：
  - `Surgical Endoscopy`
  - `Colonoscopy`
  - `Capsule Endoscopy`
  - `Gastroscopy`

这说明新数据不能只覆盖“病变描述类 query”，还必须覆盖：
- 器官/部位识别
- 解剖 landmark 识别
- 病理发现与病理类型判别
- 息肉计数
- 手术阶段/操作阶段
- 视觉定位与坐标类 query
- 质量评分类 query

但更重要的是：
- 这些 query 家族不能直接共享一套“按锚点个数递增”的标签逻辑
- 必须先判断哪些词真的能缩小检索范围，哪些只是泛场景词

## 3. 旧数据怎么分桶
基于 `/mnt/data_1/yds/多模态/data_house/origin_data/output_pairs_all_min_filtered_labeled.jsonl` 的启发式审查，建议分为三桶：

### 3.1 直接可复用
规则：
- `primary_knowledge_type` 属于 `病变特征 / 诊断评估 / 解剖特征`
- 长度适中
- 非流程说明
- 非概念讲解
- 非多图串讲

规模约：
- `13062 / 66291`

内部构成：
- `病变特征 10323`
- `解剖特征 2495`
- `诊断评估 244`

用途：
- 可直接进入“真实 query 重写”阶段
- 也可做高质量 hard-positive 种子

### 3.2 可重写复用
规则：
- 事实有价值，但文本更像描述、图说或长句说明
- 仍可保留稳定医学事实，交给 API 重写成真实 query

规模约：
- `21121 / 66291`

内部构成：
- `病变特征 16873`
- `解剖特征 3135`
- `诊断评估 1113`

用途：
- 作为主要事实供给池
- 与真实 query profile 配对后，用 API 定向生成

### 3.3 弱监督或建议丢弃
规则：
- `检查操作 / 治疗操作 / 基础概念`
- 流程图、步骤图、多图联讲、设备原理、分类图谱
- 长篇图说或强依赖图内箭头编号

规模约：
- `30156 / 66291`

用途：
- 最多保留少量做低检索价值负样本
- 不建议作为主监督来源

## 4. 新 pipeline 的核心原则

### 4.1 先围绕真实 query 建 taxonomy
不是先有锚点，再硬生成 `L0-L4`，而是：
1. 从真实测试集抽出 query 家族
2. 人工确认哪些 query 真有检索价值
3. 再反推每类 query 需要什么事实支撑

### 4.2 事实和表达解耦
- `legacy_anchor` 只负责提供事实
- `real_query_profile` 只负责提供真实表达分布
- 生成时两者配对，但标签独立判断

### 4.3 标签独立构造
- `density_level` 独立判断
- `routing_preference` 独立判断
- `retrieval_value` 单独标
- 不允许由一个标签直接映射出另一个标签

### 4.4 重新定义什么叫高密度
这里的高密度不是“词更多、槽位更多”，而是：
- 是否出现小范围限定词
- 这些限定词能否把检索空间显著缩小

有效锚点优先包括：
- 具体部位
- 具体病变
- 病理类别
- 专业名词
- 关键判别属性

默认不算有效锚点的词：
- `内镜`
- `内窥镜`
- `图片`
- `图像`
- `病变`
- `异常`
- `问题`
- `情况`

例子：
- `这张内镜图片显示什么病变`
  不是高密度，因为 `内镜/图片/病变` 都是泛词
- `小肠囊肿有什么症状`
  是高于上面的，因为 `小肠/囊肿` 是小范围检索锚点

### 4.5 重新定义 R1-R3
R 标签现在表达的是“通道倾向”，不是简单的图片依赖评级。

例子：
- `这张图片显示哪个部位`
  即使任务明确，也应是 `R3`，因为明显倾向图片通道
- `这张小肠图片显示有囊肿和病变吗`
  虽然有 `小肠/囊肿`，但文本和图片都重要，更适合 `R2`
- `小肠囊肿有什么症状`
  明显倾向文本通道，因此应是 `R1`

## 5. 推荐流程

### 阶段 A：真实 query 审查
运行：
```bash
python3 /mnt/data_1/yds/多模态/权重模块/build_rebuild_audit_packs.py
```

产物：
- `outputs/rebuild_audit/real_query_summary.v1.json`
- `outputs/rebuild_audit/real_query_manual_pack.v1.jsonl`
- `outputs/rebuild_audit/real_query_manual_pack.v1.md`

人工标注字段建议：
- `is_realistic_user_query`
- `retrieval_value`
- `density_level`
- `image_dependency`
- `rewrite_needed`
- `intent_family`

### 阶段 B：旧锚点复用审查
产物：
- `outputs/rebuild_audit/legacy_reuse_summary.v1.json`
- `outputs/rebuild_audit/legacy_reuse_manual_pack.v1.jsonl`
- `outputs/rebuild_audit/legacy_reuse_manual_pack.v1.md`

人工标注字段建议：
- `reuse_decision`: `direct_reuse / rewrite_reuse / hard_negative / discard`
- `best_use`
- `rewrite_query_example`

### 阶段 C：API 定向生成
配置文件：
- `/mnt/data_1/yds/多模态/权重模块/high_quality_data_pipeline_v2.yaml`

脚本：
- `/mnt/data_1/yds/多模态/权重模块/generate_weight_data_v2.py`

流程：
1. 读取人工确认过的 `real_query_profile`
2. 读取人工确认过的 `legacy_anchor`
3. 用 API 按 profile 重写成真实 query
4. 同时给出 `density_level / routing_preference / retrieval_value`
5. 显式列出：
   - `retrieval_anchor_terms`
   - `non_anchor_generic_terms`

### 阶段 D：独立复核
用第二个 prompt 复核：
- 是否新增事实
- 是否过于模板化
- 是否又回到了题干化表达
- `routing_preference` 是否被错误绑定

### 阶段 E：困难样本补充
单独补充四类困难样本：
- 同义改写和语序扰动
- 长但低检索价值
- 短但高检索价值
- 同样锚点、不同通道倾向

例如：
- `小肠囊肿有什么症状` -> `L较高 + R低`
- `这张小肠图片显示有囊肿吗` -> `L中等或偏高 + R高`

## 6. 质量门槛
- 同一 profile 下不得批量生成近似模板
- `R1-R3` 标签分布不能再与 `density_level` 一一映射
- 泛词不能被批量误记为高价值检索锚点
- query 中带强指代表达但无医学实体的样本比例必须受控
- 每一大类真实任务至少覆盖 8 个以上 profile

## 7. 这版方案和旧方案最本质的区别
- 旧方案：从描述文本出发，靠模板递增信息量
- 新方案：从真实 query 分布出发，向后匹配可支撑的事实锚点

一句话说：
旧方案像“把图说改成问句”，
新方案是“先知道真实用户怎么问，再挑哪些事实值得被问、该怎么问”。
