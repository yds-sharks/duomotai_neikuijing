#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List


OUT_DIR = Path("/mnt/data_1/yds/多模态/权重模块/人工审核/06_人工标注输入")
JSONL_OUT = OUT_DIR / "alignment_case_pack.v1.jsonl"
MD_OUT = OUT_DIR / "alignment_case_pack.v1.md"


REAL_QUERY_CASES: List[Dict] = [
    {
        "sample_id": "rq_001",
        "source_type": "real_query",
        "task_family": "organ_identification",
        "text": "请判断此内窥镜图像中显示的是哪个器官。",
        "options_text": "A: 小肠\nB: 大肠\nC: 食管\nD: 胃",
        "suspected_anchor_terms": [],
        "suspected_generic_terms": ["内窥镜图像", "器官"],
        "ambiguity_reason": "任务目标明确，但文本几乎没有小范围检索锚点；很容易被误判成中高密度。",
    },
    {
        "sample_id": "rq_002",
        "source_type": "real_query",
        "task_family": "organ_identification",
        "text": "该内窥镜图像中显示的是哪个解剖结构？",
        "options_text": "A: 大肠\nB: 胃\nC: 食管\nD: 小肠",
        "suspected_anchor_terms": [],
        "suspected_generic_terms": ["内窥镜图像", "解剖结构"],
        "ambiguity_reason": "“解剖结构”是不是有效锚点边界不稳，需要人工钉住。",
    },
    {
        "sample_id": "rq_003",
        "source_type": "real_query",
        "task_family": "lesion_type",
        "text": "分析此内窥镜图像以识别任何病理发现。如有发现，应归入何种分类？",
        "options_text": "A: 胃息肉\nB: 结直肠癌\nC: 巴雷特食管\nD: 食管静脉曲张",
        "suspected_anchor_terms": [],
        "suspected_generic_terms": ["内窥镜图像", "病理发现", "分类"],
        "ambiguity_reason": "容易因为“病理”“分类”这些看起来专业的词被误判为高密度。",
    },
    {
        "sample_id": "rq_004",
        "source_type": "real_query",
        "task_family": "lesion_type",
        "text": "该图像是否显示任何病理发现？如果有，请描述所识别异常的性质。",
        "options_text": "A: 血管扩张\nB: 巴雷特食管\nC: 胃息肉\nD: 无异常",
        "suspected_anchor_terms": [],
        "suspected_generic_terms": ["图像", "病理发现", "异常", "性质"],
        "ambiguity_reason": "典型高图片依赖，但文本表面有医学味，容易误标。",
    },
    {
        "sample_id": "rq_005",
        "source_type": "real_query",
        "task_family": "quality_score",
        "text": "该内镜图像所示的波士顿肠道准备量表（BBPS）评分是多少？",
        "options_text": "A: BBPS 0-1\nB: BBPS 2-3",
        "suspected_anchor_terms": ["BBPS"],
        "suspected_generic_terms": ["内镜图像", "评分"],
        "ambiguity_reason": "有专业名词 BBPS，但最终判断高度依赖图像；适合校准高L高R是否允许同时存在。",
    },
    {
        "sample_id": "rq_006",
        "source_type": "real_query",
        "task_family": "quality_score",
        "text": "根据内镜图像中的发现，如何评估波士顿肠道准备量表（BBPS）评分？",
        "options_text": "A: BBPS 0-1\nB: BBPS 2-3",
        "suspected_anchor_terms": ["BBPS"],
        "suspected_generic_terms": ["内镜图像", "发现", "评分"],
        "ambiguity_reason": "比 rq_005 更像描述性问法，适合和 rq_005 做成一对对齐样本。",
    },
    {
        "sample_id": "rq_007",
        "source_type": "real_query",
        "task_family": "visual_grounding",
        "text": "请识别内镜图像中显示的高级腺瘤的坐标。回答格式要求为[x1, y1, x2, y2]。",
        "options_text": "A: [478, 31, 1225, 810]\nB: [560, 231, 1017, 931]\nC: [0, 287, 594, 1079]\nD: [324, 95, 976, 826]",
        "suspected_anchor_terms": ["高级腺瘤"],
        "suspected_generic_terms": ["内镜图像", "坐标"],
        "ambiguity_reason": "有具体病变名，但任务是纯图像定位，不一定有高检索价值。",
    },
    {
        "sample_id": "rq_008",
        "source_type": "real_query",
        "task_family": "severity_score",
        "text": "从此内镜图像中呈现的发现可以推断出溃疡性结肠炎的Mayo评分是多少？",
        "options_text": "A: UCG 0-1\nB: UCG 1\nC: UCG 1-2\nD: UCG 2",
        "suspected_anchor_terms": ["溃疡性结肠炎", "Mayo评分"],
        "suspected_generic_terms": ["内镜图像", "发现"],
        "ambiguity_reason": "文本存在明确病种和评分体系，但是否应算高密度需要人工定边界。",
    },
    {
        "sample_id": "rq_009",
        "source_type": "synthetic_boundary_query",
        "task_family": "dependency_boundary",
        "text": "这张小肠图片显示有囊肿和病变吗",
        "options_text": "",
        "suspected_anchor_terms": ["小肠", "囊肿"],
        "suspected_generic_terms": ["图片", "病变"],
        "ambiguity_reason": "这是你明确提出的关键边界例子，必须作为 few-shot 固定下来。",
    },
    {
        "sample_id": "rq_010",
        "source_type": "synthetic_boundary_query",
        "task_family": "dependency_boundary",
        "text": "小肠囊肿有什么症状",
        "options_text": "",
        "suspected_anchor_terms": ["小肠", "囊肿"],
        "suspected_generic_terms": [],
        "ambiguity_reason": "与 rq_009 构成同锚点不同图片依赖的强对照样本。",
    },
    {
        "sample_id": "rq_011",
        "source_type": "synthetic_boundary_query",
        "task_family": "dependency_boundary",
        "text": "这张图片显示哪个部位",
        "options_text": "",
        "suspected_anchor_terms": [],
        "suspected_generic_terms": ["图片", "部位"],
        "ambiguity_reason": "高图片依赖、低检索密度的核心负例。",
    },
    {
        "sample_id": "rq_012",
        "source_type": "synthetic_boundary_query",
        "task_family": "density_boundary",
        "text": "胃部有异常吗",
        "options_text": "",
        "suspected_anchor_terms": ["胃部"],
        "suspected_generic_terms": ["异常"],
        "ambiguity_reason": "只有一个较粗部位词，适合用来钉 L1/L2 的边界。",
    },
]


LEGACY_CASES: List[Dict] = [
    {
        "sample_id": "lg_001",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "病变特征",
        "text": "胸部CT图像显示纵隔气肿，箭头所指区域为气肿部位，该情况发生于ESD术后，提示可能存在食管穿孔。",
        "suspected_anchor_terms": ["纵隔气肿", "ESD术后", "食管穿孔"],
        "suspected_generic_terms": ["CT图像", "箭头所指", "该情况"],
        "ambiguity_reason": "事实强，但原文仍带图像指代；需要人工决定重写后应更偏文本检索还是图片判断。",
    },
    {
        "sample_id": "lg_002",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "诊断评估",
        "text": "保守治疗1周后进行胃部造影，结果显示造影剂在胃内均匀分布，未观察到造影剂外溢，提示胃壁无穿孔。",
        "suspected_anchor_terms": ["胃部造影", "造影剂外溢", "胃壁无穿孔"],
        "suspected_generic_terms": ["结果显示", "提示"],
        "ambiguity_reason": "事实完整，但像病例描述，不确定最适合生成解释型 query 还是检索型 query。",
    },
    {
        "sample_id": "lg_003",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "病变特征",
        "text": "食管与胃连接处的溃疡病变观察不完整",
        "suspected_anchor_terms": ["食管与胃连接处", "溃疡"],
        "suspected_generic_terms": ["病变", "观察不完整"],
        "ambiguity_reason": "非常短，但信息不稳定，适合校准短文本是否一定低密度。",
    },
    {
        "sample_id": "lg_004",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "病变特征",
        "text": "在胃体上部小弯处观察到0~IIc型病变，病变区域黏膜表面呈现不规则隆起，颜色略显红色，局部伴有轻微充血和糜烂表现。",
        "suspected_anchor_terms": ["胃体上部小弯", "0~IIc型", "不规则隆起", "充血", "糜烂"],
        "suspected_generic_terms": ["病变区域"],
        "ambiguity_reason": "典型高价值事实，适合作为高 L 样本，但需要人工决定是否应该带图片依赖问法版本。",
    },
    {
        "sample_id": "lg_005",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "病变特征",
        "text": "内镜图像显示食管黏膜表面存在高度扩张的不规则血管，伴有局部出血和暗红色血凝块，红色箭头指向该异常血管区域，其表现符合类型B3的特征。",
        "suspected_anchor_terms": ["食管黏膜", "不规则血管", "局部出血", "血凝块", "类型B3"],
        "suspected_generic_terms": ["内镜图像", "箭头指向", "异常血管区域"],
        "ambiguity_reason": "有很多有效锚点，但原文强图像化；适合构造一对 R 低 / R 高 对照样本。",
    },
    {
        "sample_id": "lg_006",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "病变特征",
        "text": "内镜图像显示消化道黏膜表面，数个白色箭头指向的病变区域提示为EP、LPM深度的0~Ⅱb型病变。",
        "suspected_anchor_terms": ["EP", "LPM", "0~Ⅱb型"],
        "suspected_generic_terms": ["内镜图像", "病变区域", "箭头指向"],
        "ambiguity_reason": "既有强专业锚点，也有很强图像依赖信号，是高价值对齐样本。",
    },
    {
        "sample_id": "lg_007",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "病变特征",
        "text": "内镜图像显示胃黏膜表面存在病变区域，该区域表现为不规则隆起，表面可见白色渗出物及血管纹理。箭头所指位置为SM2深度的病变。根据有马分类，该病变属于类型4；根据井上分类，该病变属于类型V-N。",
        "suspected_anchor_terms": ["胃黏膜", "不规则隆起", "白色渗出物", "SM2", "有马分类4", "井上分类V-N"],
        "suspected_generic_terms": ["内镜图像", "病变区域", "箭头所指"],
        "ambiguity_reason": "高信息密度，但重写时很容易过度保留图像指代，需要人工给出理想版本。",
    },
    {
        "sample_id": "lg_008",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "解剖特征",
        "text": "口咽部结构显示悬雍垂位于中央，软腭位于悬雍垂的下方，咽腭弓和舌腭弓分别位于两侧。",
        "suspected_anchor_terms": ["口咽部", "悬雍垂", "软腭", "咽腭弓", "舌腭弓"],
        "suspected_generic_terms": ["结构显示"],
        "ambiguity_reason": "解剖事实明确，但不一定天然对应高检索价值 query，需要人工判用途。",
    },
    {
        "sample_id": "lg_009",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "病变特征",
        "text": "图像显示褐色区域，其内可见扩张、蜷曲、直径不一、形状不同的异型血管，这些血管与周围黏膜有明确边界，且病变累及全周，提示存在癌性病灶。",
        "suspected_anchor_terms": ["异型血管", "扩张", "蜷曲", "直径不一", "癌性病灶"],
        "suspected_generic_terms": ["图像显示", "病变"],
        "ambiguity_reason": "适合测试模型是否会把多个属性词真正识别为小范围锚点。",
    },
    {
        "sample_id": "lg_010",
        "source_type": "legacy_anchor",
        "primary_knowledge_type": "病变特征",
        "text": "皮肤表面呈现密集的点状色素沉着，颜色为深褐色至黑色，分布均匀，背景为浅粉色皮肤组织，左上角标有字母'A'。根据有马分类系统，该表现属于类型3a；根据井上分类系统，属于类型V-1，提示病变累及M1和M2层。",
        "suspected_anchor_terms": ["点状色素沉着", "类型3a", "类型V-1", "M1", "M2"],
        "suspected_generic_terms": ["左上角标有字母A", "提示病变"],
        "ambiguity_reason": "包含可能跨域的混杂描述，适合人工判定是否应直接丢弃。",
    },
]


def build_rows() -> List[Dict]:
    rows = []
    for item in REAL_QUERY_CASES + LEGACY_CASES:
        row = dict(item)
        row["annotation_template"] = {
            "density_level": "",
            "image_dependency": "",
            "retrieval_value": "",
            "retrieval_anchor_terms": [],
            "generic_non_anchor_terms": [],
            "rewrite_query_example": "",
            "is_good_few_shot_case": "",
            "notes": "",
        }
        rows.append(row)
    return rows


def write_jsonl(rows: List[Dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL_OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_md(rows: List[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Alignment Case Pack V1")
    lines.append("")
    lines.append("用途：")
    lines.append("- 人工对齐 L/R 标签边界")
    lines.append("- 选出适合做 few-shot 的案例")
    lines.append("- 后续作为 API 生成时的上下文提示")
    lines.append("")
    for idx, row in enumerate(rows, start=1):
        lines.append(f"## Case {idx}")
        lines.append(f"- sample_id: {row['sample_id']}")
        lines.append(f"- source_type: {row['source_type']}")
        if row.get("task_family"):
            lines.append(f"- task_family: {row['task_family']}")
        if row.get("primary_knowledge_type"):
            lines.append(f"- primary_knowledge_type: {row['primary_knowledge_type']}")
        lines.append(f"- text: {row['text']}")
        if row.get("options_text"):
            lines.append("- options_text:")
            lines.append("```text")
            lines.append(row["options_text"])
            lines.append("```")
        lines.append(f"- suspected_anchor_terms: {row['suspected_anchor_terms']}")
        lines.append(f"- suspected_generic_terms: {row['suspected_generic_terms']}")
        lines.append(f"- ambiguity_reason: {row['ambiguity_reason']}")
        lines.append("- annotation_template:")
        lines.append("```json")
        lines.append(json.dumps(row["annotation_template"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_jsonl(rows)
    write_md(rows)
    print(f"[done] {JSONL_OUT}")
    print(f"[done] {MD_OUT}")


if __name__ == "__main__":
    main()
