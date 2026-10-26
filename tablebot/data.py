"""读取题目清单，修正已知的数据问题，把题目里的文件名对应到真实文件。"""
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

VALID_TYPES = ("structure", "extract", "thinking")
VALID_FORMATS = ("json", "json_array", "number", "string")

# 用来给 question_type 缺失/错误的行推断题型
_THINK_WORDS = ("计算", "合计", "总和", "求和", "差值", "插值", "比例", "占比", "增长", "最高", "最低",
                "最大", "最小", "排序", "多少个", "几个", "几类", "几天", "哪些天", "判断", "平均",
                "calculate", "sum", "total", "average", "highest", "lowest")


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v).strip()


def _infer_type(question: str, answer_format: str) -> str:
    if answer_format == "json" or "恢复" in question:
        return "structure"
    q = question.lower()
    if any(w in q for w in _THINK_WORDS):
        return "thinking"
    return "extract"


class FileResolver:
    """题目中的文件名有时和真实文件不一致（如 58.pdf / 0060.pdf → 058.pdf / 060.pdf）。"""

    def __init__(self, files_dir: Path):
        self.files_dir = files_dir
        self.by_name: Dict[str, Path] = {}
        self.by_num: Dict[Tuple[int, str], Path] = {}
        for p in files_dir.iterdir():
            if not p.is_file():
                continue
            self.by_name[p.name.lower()] = p
            if p.stem.isdigit():
                self.by_num[(int(p.stem), p.suffix.lower())] = p

    def resolve(self, name: str) -> Optional[Path]:
        name = name.strip()
        p = self.by_name.get(name.lower())
        if p:
            return p
        stem, _, ext = name.rpartition(".")
        m = re.search(r"\d+", stem)
        if m:
            return self.by_num.get((int(m.group()), "." + ext.lower()))
        return None


def load_questions(tests_path: Path, files_dir: Path) -> List[dict]:
    df = pd.read_excel(tests_path, dtype=object)
    resolver = FileResolver(files_dir)
    out = []
    for _, r in df.iterrows():
        qid = int(float(_clean(r.get("id"))))
        q = {
            "id": qid,
            "file_name": _clean(r.get("file_name")),
            "question_type": _clean(r.get("question_type")),
            "question": _clean(r.get("question")),
            "table_hint": _clean(r.get("table_hint")),
            "answer_format": _clean(r.get("answer_format")).lower(),
            "fixes": [],
        }
        if q["answer_format"] not in VALID_FORMATS:
            q["fixes"].append(f"answer_format={q['answer_format']!r} 无效，按 string 处理")
            q["answer_format"] = "string"
        if q["question_type"] not in VALID_TYPES:
            # 例如 id=63：question_type 列里填的是题目文本
            if not q["question"]:
                q["question"] = q["question_type"]
            new = _infer_type(q["question"], q["answer_format"])
            q["fixes"].append(f"question_type={q['question_type'][:20]!r} 无效，推断为 {new}")
            q["question_type"] = new
        path = resolver.resolve(q["file_name"])
        q["path"] = str(path) if path else ""
        if path is None:
            q["fixes"].append(f"找不到文件 {q['file_name']}")
        elif path.name != q["file_name"]:
            q["fixes"].append(f"文件名 {q['file_name']} → {path.name}")
        out.append(q)
    return out


def parse_id_spec(spec: str) -> List[int]:
    """'1,3,10-20' → [1,3,10,...,20]"""
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids
