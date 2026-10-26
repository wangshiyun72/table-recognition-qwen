"""解析模型输出 + 按答案规范清洗。评分是全对或全错，这一步非常关键。"""
import ast
import json
import operator
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------- JSON 解析

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_loose(text: str) -> Any:
    """从模型输出中尽量解析出 JSON（容忍 ```json 包裹、前后多余文字、尾逗号）。"""
    if text is None:
        raise ValueError("空输出")
    s = text.strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    for cand in (s, _outermost(s, "{", "}"), _outermost(s, "[", "]")):
        if not cand:
            continue
        for fixed in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                return json.loads(fixed, parse_float=Decimal)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"无法解析 JSON：{text[:200]!r}")


def salvage_answer(text: str) -> Any:
    """整体 JSON 坏掉时（比如 evidence 里引号没转义），单独把 "answer": 后面的值解析出来。"""
    dec = json.JSONDecoder(parse_float=Decimal)
    for m in reversed(list(re.finditer(r'"answer"\s*:\s*', text or ""))):
        try:
            val, _ = dec.raw_decode(text, m.end())
            return {"answer": val, "_salvaged": True}
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法解析 JSON：{(text or '')[:200]!r}")


def _outermost(s: str, lo: str, hi: str) -> Optional[str]:
    i, j = s.find(lo), s.rfind(hi)
    return s[i:j + 1] if 0 <= i < j else None


# ---------------------------------------------------------------- 数字

_NUM_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?%?$|^[+-]?\d+(\.\d+)?%?$")
_CURRENCY = "¥￥$€£"


NUM_TYPES = (int, float, Decimal)


def _num_to_str(x) -> str:
    if isinstance(x, Decimal):
        return format(x, "f")
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        d = Decimal(repr(x))
        return format(d, "f")
    return str(x)


def clean_number(v: Any) -> str:
    """number 型答案：去千分位、货币符号和空格；保留小数位、负号、%。"""
    if v is None:
        return ""
    if isinstance(v, NUM_TYPES) and not isinstance(v, bool):
        return _num_to_str(v)
    s = str(v).strip()
    s = s.replace("，", ",").replace("．", ".").replace("％", "%").replace("−", "-").replace("－", "-")
    s = s.strip(_CURRENCY).strip()
    s = re.sub(r"\s+", "", s)
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?%?", s):      # 千分位：1,234,567.89
        s = s.replace(",", "")
    elif re.fullmatch(r"[+-]?\d+,\d{1,2}%?", s):                 # 小数逗号（欧洲/巴西写法）：9,4 → 9.4
        s = s.replace(",", ".")
    return s


def _is_plain_number(s: str) -> bool:
    return bool(_NUM_RE.match(s))


# ---------------------------------------------------------------- 算式复核

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> Optional[float]:
    if not expr or not isinstance(expr, str):
        return None
    e = expr.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")")
    e = re.sub(r"(?<=\d),(?=\d{3})", "", e)      # 去千分位
    e = e.replace("%", "")
    if "=" in e:
        e = e.split("=")[0]
    if not re.fullmatch(r"[\d\s.+\-*/()]+", e):
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("不支持的表达式")

    try:
        return float(ev(ast.parse(e, mode="eval")))
    except Exception:  # noqa: BLE001
        return None


def _decimals(s: str) -> int:
    m = re.search(r"\.(\d+)", s)
    return len(m.group(1)) if m else 0


def reconcile_number(answer: str, expr_value: Optional[float]) -> Tuple[str, Optional[str]]:
    """用程序算出的结果复核模型给的数值，格式（小数位、%）沿用模型答案。返回 (答案, 备注)。"""
    if expr_value is None:
        return answer, None
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(%?)", answer)
    if not m:
        return answer, None
    model_val, pct = float(m.group(1)), m.group(2)
    candidates = [expr_value]
    if pct:
        candidates.append(expr_value * 100)  # 算式给的是小数比值
    best = min(candidates, key=lambda c: abs(c - model_val))
    nd = _decimals(answer)
    tol = 0.5 * 10 ** (-nd) + 1e-9
    if abs(best - model_val) <= tol:
        return answer, None  # 一致（在四舍五入误差内）
    if nd == 0 and not float(best).is_integer():
        nd = 2
    fixed = f"{best:.{nd}f}"
    if "." in fixed and nd > 0 and _decimals(answer) == 0:
        fixed = fixed.rstrip("0").rstrip(".")
    return fixed + pct, f"算式复核：模型答 {answer}，算式得 {fixed + pct}"


# ---------------------------------------------------------------- 文本

_PREFIX = re.compile(r"^(答案(是|为)?[:：]?|根据表格可知[，,:：]?|最终答案[:：]?)\s*")


def clean_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, NUM_TYPES) and not isinstance(v, bool):
        return _num_to_str(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, default=_num_to_str)
    s = str(v).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"[ \t　]+", " ", s).strip()
    s = _PREFIX.sub("", s)
    # 中文字符之间的空格去掉
    s = re.sub(r"(?<=[一-鿿]) (?=[一-鿿])", "", s)
    return s


def clean_array(v: Any, numeric_as_number: bool = True) -> str:
    """json_array 型答案。纯数字元素输出为 JSON 数字，并保留表格中的小数位（如 1234.00）。"""
    if v is None or v == "":
        return "[]"
    if isinstance(v, str):
        try:
            v = parse_json_loose(v)
        except ValueError:
            v = [v]
    if not isinstance(v, list):
        v = [v]
    parts = []
    for x in v:
        if isinstance(x, (list, dict)):
            parts.append(json.dumps(x, ensure_ascii=False, default=_num_to_str))
        elif x is None:
            parts.append('""')
        elif isinstance(x, bool):
            parts.append(json.dumps(x))
        elif isinstance(x, NUM_TYPES):
            parts.append(_num_to_str(x) if numeric_as_number else json.dumps(_num_to_str(x)))
        else:
            s = clean_text(x)
            c = clean_number(s)
            if numeric_as_number and _is_plain_number(c) and not c.endswith("%"):
                c = c.lstrip("+")
                if re.fullmatch(r"-?(0|[1-9]\d*)(\.\d+)?", c):   # 合法 JSON 数字（无前导 0）
                    parts.append(c)
                    continue
            parts.append(json.dumps(c if _is_plain_number(c) else s, ensure_ascii=False))
    return "[" + ", ".join(parts) + "]"


# ---------------------------------------------------------------- 结构题
#
# 做法：不管题目问哪一部分，都让模型输出【整张表】的全部单元格 + 表头行数，
# 再由程序按题目文字截取范围。模型自己截取范围时很容易出错（漏掉第一列、把数据行当表头等）。

_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_NUM_PAT = r"([0-9]+|[一二两三四五六七八九十]+)"


def _to_int(tok: str) -> int:
    if tok.isdigit():
        return int(tok)
    if tok == "十":
        return 10
    if tok.startswith("十"):
        return 10 + _CN_NUM.get(tok[1:], 0)
    if "十" in tok:
        a, _, b = tok.partition("十")
        return _CN_NUM.get(a, 1) * 10 + (_CN_NUM.get(b, 0) if b else 0)
    return _CN_NUM.get(tok, 1)


def parse_scope(question: str) -> dict:
    """从题目文字解析恢复范围。返回 {"all": bool, "rows": set|None, "cols": set|None, "header": bool}"""
    q = question.replace(" ", "")
    scope = {"all": False, "rows": None, "cols": None, "header": False}
    if re.search(r"整个表|整张表|全部|完整|整表", q):
        scope["all"] = True
        return scope
    for unit, key in (("行", "rows"), ("列", "cols")):
        idx = set()
        for m in re.finditer(r"(前|第)" + _NUM_PAT + unit, q):
            n = _to_int(m.group(2))
            idx |= set(range(n)) if m.group(1) == "前" else {n - 1}
        # “第1-3行”
        for m in re.finditer(r"第" + _NUM_PAT + r"[-~至到]" + _NUM_PAT + unit, q):
            idx |= set(range(_to_int(m.group(1)) - 1, _to_int(m.group(2))))
        if idx:
            scope[key] = idx
    if "表头" in q:
        scope["header"] = True
    if scope["rows"] is None and scope["cols"] is None and not scope["header"]:
        scope["all"] = True
    return scope


def _cell_from(c) -> Optional[dict]:
    """支持两种写法：{"text","row","col","rowspan","colspan"} 或 [row, col, rowspan, colspan, text]"""
    if isinstance(c, (list, tuple)) and len(c) >= 5:
        r, col, rs, cs, text = c[:5]
    elif isinstance(c, dict):
        r, col = c.get("row", 0), c.get("col", 0)
        rs, cs, text = c.get("rowspan", 1), c.get("colspan", 1), c.get("text", "")
    else:
        return None
    try:
        return {"text": clean_text(text), "row": int(r), "col": int(col),
                "rowspan": max(1, int(rs or 1)), "colspan": max(1, int(cs or 1))}
    except (TypeError, ValueError):
        return None


def _drop_duplicate_rows(cells: list, rc: int) -> Tuple[list, int, Optional[int]]:
    rows = {}
    for c in cells:
        rows.setdefault(c["row"], []).append(c)
    for r in sorted(rows):
        if r + 1 not in rows:
            continue
        a = [(c["col"], c["text"]) for c in rows[r] if c["text"]]
        b = [(c["col"], c["text"]) for c in rows[r + 1] if c["text"]]
        ta, tb = [t for _, t in a], [t for _, t in b]
        if len(ta) >= 2 and len(tb) >= 2:
            same = len(set(ta) & set(tb)) / max(len(set(ta)), len(set(tb)))
            if same >= 0.8:
                out = []
                for c in cells:
                    if c["row"] == r + 1:
                        continue
                    if c["row"] > r + 1:
                        c = dict(c, row=c["row"] - 1)
                    elif c["row"] <= r < c["row"] + c["rowspan"] - 1:
                        c = dict(c, rowspan=c["rowspan"] - 1)
                    out.append(c)
                return out, max(0, rc - 1), r + 1
    return cells, rc, None


def _repair_overlaps(cells: list) -> Tuple[list, list]:
    notes = []
    for _ in range(5):  # 迭代几轮直到无冲突
        owner, changed, kept = {}, False, []
        for c in cells:
            o = owner.get((c["row"], c["col"]))
            if o is not None:
                if o["row"] < c["row"]:
                    o["rowspan"] = c["row"] - o["row"]
                    notes.append(f"({o['row']},{o['col']}) 跨行数缩小为 {o['rowspan']}")
                    changed = True
                    break
                if o["col"] < c["col"]:
                    o["colspan"] = c["col"] - o["col"]
                    notes.append(f"({o['row']},{o['col']}) 跨列数缩小为 {o['colspan']}")
                    changed = True
                    break
                notes.append(f"({c['row']},{c['col']}) 重复，已丢弃")
                continue
            for i in range(c["row"], c["row"] + c["rowspan"]):
                for j in range(c["col"], c["col"] + c["colspan"]):
                    owner.setdefault((i, j), c)
            kept.append(c)
        if not changed:
            return kept, notes
    return cells, notes


def clean_structure(obj: Any, question: str = "") -> Tuple[str, list]:
    """校验整表结构 → 按题目截取范围 → 输出比赛要求的 JSON 字符串。返回 (答案, 备注)。"""
    warns = []
    if isinstance(obj, str):
        obj = parse_json_loose(obj)
    if isinstance(obj, dict) and "cells" not in obj and isinstance(obj.get("answer"), (dict, str)):
        obj = obj["answer"]
        if isinstance(obj, str):
            obj = parse_json_loose(obj)
    if not isinstance(obj, dict):
        raise ValueError("结构题输出不是 JSON 对象")

    cells, seen = [], set()
    for raw in obj.get("cells") or []:
        c = _cell_from(raw)
        if c is None or c["row"] < 0 or c["col"] < 0:
            warns.append(f"无效单元格 {str(raw)[:40]}")
            continue
        if (c["row"], c["col"]) in seen:
            warns.append(f"重复单元格 ({c['row']},{c['col']})")
            continue
        seen.add((c["row"], c["col"]))
        cells.append(c)
    cells.sort(key=lambda c: (c["row"], c["col"]))

    def _int(x, default=0):
        try:
            return int(x)
        except (TypeError, ValueError):
            return default

    # 行列数：取模型给的值与单元格实际占用的最大值
    rc, cc = _int(obj.get("row_count")), _int(obj.get("col_count"))
    need_r = max((c["row"] + c["rowspan"] for c in cells), default=0)
    need_c = max((c["col"] + c["colspan"] for c in cells), default=0)
    if rc < need_r:
        warns.append(f"row_count {rc} < 单元格实际占用 {need_r}，已修正")
        rc = need_r
    if cc < need_c:
        warns.append(f"col_count {cc} < 单元格实际占用 {need_c}，已修正")
        cc = need_c

    # 重复行：模型偶尔把表头整行重复输出一遍（相邻两行文本几乎完全相同），删掉后一行并上移
    cells, rc, dup_row = _drop_duplicate_rows(cells, rc)
    if dup_row is not None:
        warns.append(f"第 {dup_row} 行与上一行重复，已删除")
        hr = _int(obj.get("header_rows"), 0)
        if dup_row < hr:
            obj = dict(obj, header_rows=max(1, hr - 1))

    # 合并冲突：某单元格的左上角落在前面某个合并区域里，通常是前者的跨行/跨列数估大了，
    # 把前者的跨度缩小到不冲突；左上角完全重复的才丢弃
    cells, fix_notes = _repair_overlaps(cells)
    warns.extend(fix_notes)

    # 按题目截取范围
    scope = parse_scope(question)
    if not scope["all"]:
        header_rows = _int(obj.get("header_rows"), 0)
        if header_rows <= 0:  # 兜底：第 0 行最大的 rowspan
            header_rows = max((c["rowspan"] for c in cells if c["row"] == 0), default=1)
        rows = set(scope["rows"] or set())
        if scope["header"]:
            rows |= set(range(header_rows))
        cols = scope["cols"] or set()
        cells = [c for c in cells if c["row"] in rows or c["col"] in cols]
        warns.append(f"按题目截取：行{sorted(rows)[:10]} 列{sorted(cols)[:10]}（表头行数 {header_rows}）")

    out = {"row_count": rc, "col_count": cc, "cells": cells}
    return json.dumps(out, ensure_ascii=False), warns[:20]


# ---------------------------------------------------------------- 模板题后处理

_NEG = ("否", "不", "无", "没", "未", "非", "false", "no", "none", "n/a")
_POS_WORDS = {"是", "true", "yes", "y", "对", "正确", "有", "存在", "包含", "含"}


def to_yes_no(v: Any) -> str:
    """把“是否”类题目的答案统一成“是”/“否”。字段内容（非空且非否定）视为“是”。"""
    if isinstance(v, bool):
        return "是" if v else "否"
    s = clean_text(v).strip().strip("。.").lower()
    if s in ("是", "否"):
        return s
    if not s:
        return "否"
    if s in _POS_WORDS:
        return "是"
    if s.startswith(_NEG) or s in ("0",):
        return "否"
    return "是"


_LANG_MAP = {
    "english": "英文", "英语": "英文", "chinese": "中文", "汉语": "中文", "简体中文": "中文", "繁体中文": "中文",
    "spanish": "西班牙语", "español": "西班牙语", "espanol": "西班牙语",
    "portuguese": "葡萄牙语", "português": "葡萄牙语", "arabic": "阿拉伯语", "lithuanian": "立陶宛语",
    "lietuvių": "立陶宛语", "norwegian": "挪威语", "norsk": "挪威语", "slovak": "斯洛伐克语",
    "slovenčina": "斯洛伐克语", "dutch": "荷兰语", "nederlands": "荷兰语", "french": "法语", "français": "法语",
    "german": "德语", "deutsch": "德语", "japanese": "日语", "korean": "韩语", "russian": "俄语",
    "italian": "意大利语", "czech": "捷克语", "polish": "波兰语", "turkish": "土耳其语",
}


_LANG_KEYWORDS = [  # (关键词, 标准名)，按常见程度排列
    (("中文", "汉语", "chinese", "简体", "繁体"), "中文"),
    (("英文", "英语", "english"), "英文"),
    (("阿拉伯", "arabic"), "阿拉伯语"), (("西班牙", "spanish", "español", "espanol"), "西班牙语"),
    (("葡萄牙", "portuguese", "português"), "葡萄牙语"), (("立陶宛", "lithuanian", "lietuvių"), "立陶宛语"),
    (("挪威", "norwegian", "norsk"), "挪威语"), (("斯洛伐克", "slovak", "slovenčina"), "斯洛伐克语"),
    (("荷兰", "dutch", "nederlands"), "荷兰语"), (("法语", "法文", "french", "français"), "法语"),
    (("德语", "德文", "german", "deutsch"), "德语"), (("俄语", "俄文", "russian"), "俄语"),
    (("日语", "日文", "japanese"), "日语"), (("韩语", "韩文", "korean"), "韩语"),
    (("意大利", "italian"), "意大利语"), (("捷克", "czech"), "捷克语"), (("波兰", "polish"), "波兰语"),
]


def normalize_language(v: Any) -> str:
    """语言题：从答案中识别出提到的语言，统一成“中文”“英文”“中文和英文”这类写法。"""
    if isinstance(v, list):
        v = " ".join(str(x) for x in v)
    s = clean_text(v)
    low = s.lower()
    found = []
    for keys, name in _LANG_KEYWORDS:
        pos = [low.find(k) for k in keys if k in low]
        if pos:
            found.append((min(pos), name))
    if "中英" in s or "双语" in s:
        found += [(-2, "中文"), (-1, "英文")]
    names = list(dict.fromkeys(n for _, n in sorted(found)))
    if not names:
        return s
    if set(names) >= {"中文", "英文"}:
        rest = [n for n in names if n not in ("中文", "英文")]
        return "和".join(["中文", "英文"] + rest)
    return "和".join(names)


def _accounting_number(s: str) -> str:
    """number 型：会计负数 (100) → -100；“5045(33.1)” 这类取括号前的主数值。"""
    m = re.fullmatch(r"\(([\d,]+(?:\.\d+)?)\)", s)
    if m:
        return "-" + m.group(1).replace(",", "")
    m = re.fullmatch(r"([+-]?[\d,]+(?:\.\d+)?%?)\s*[\(（].*[\)）]", s)
    if m:
        return clean_number(m.group(1))
    return s


# ---------------------------------------------------------------- 总入口

def finalize(q: dict, content: str, numeric_as_number: bool = True,
             trust_expression: bool = True) -> dict:
    """把模型原始输出变成最终 answer 字符串。"""
    fmt = q["answer_format"]
    res = {"answer": "", "notes": [], "parsed": None}
    is_struct = q["question_type"] == "structure"
    try:
        obj = parse_json_loose(content)
    except ValueError:
        if is_struct:
            raise
        obj = salvage_answer(content)
        res["notes"].append("JSON 不完整，已单独提取 answer 字段")

    if is_struct:
        ans, warns = clean_structure(obj, q.get("question", ""))
        res["answer"], res["notes"] = ans, warns
        return res

    ans = obj.get("answer", obj) if isinstance(obj, dict) else obj
    qtext = q.get("question", "")
    is_yes_no = "是否" in qtext and fmt == "string"
    if fmt == "json_array" and isinstance(ans, list):
        # 数组里的布尔值统一写成“是/否”
        ans = [("是" if x else "否") if isinstance(x, bool) else
               (to_yes_no(x) if isinstance(x, str) and x.strip().lower() in ("true", "false") else x)
               for x in ans]
    if is_yes_no:
        new = to_yes_no(ans)
        if clean_text(ans) != new:
            res["notes"].append(f"是否题：{clean_text(ans)[:30]} → {new}")
        res["answer"] = new
        return res
    if re.search(r"使用哪种语言|语言或语言组合", qtext) and fmt == "string":
        res["answer"] = normalize_language(ans)
        return res
    if fmt == "number" and "列数" in qtext and isinstance(obj, dict) and isinstance(obj.get("columns"), list) \
            and obj["columns"]:
        # 列数题：以模型逐列列出的列名个数为准，避免目测数错
        n = len(obj["columns"])
        if clean_number(ans) != str(n):
            res["notes"].append(f"列数按逐列清单计：模型答 {ans}，清单 {n} 列")
        res["answer"] = str(n)
        return res
    if fmt == "number":
        if isinstance(ans, list) and len(ans) == 1:
            ans = ans[0]
        s = clean_number(clean_text(ans) if not isinstance(ans, NUM_TYPES) else ans)
        s = _accounting_number(s)
        if trust_expression and q["question_type"] == "thinking" and isinstance(obj, dict):
            s, note = reconcile_number(s, safe_eval(obj.get("expression", "")))
            if note:
                res["notes"].append(note)
        res["answer"] = s
    elif fmt == "json_array":
        res["answer"] = clean_array(ans, numeric_as_number)
    else:
        if isinstance(ans, list):
            ans = "、".join(clean_text(x) for x in ans) if len(ans) != 1 else ans[0]
        t = clean_text(ans)
        if _is_plain_number(clean_number(t)) and re.search(r"\d,\d{3}", t):
            t = clean_number(t)  # 纯数字字符串也要去千分位
        res["answer"] = t
    return res
