"""
复杂表格识别与内容理解 —— baseline 入口

常用命令（在项目根目录执行）：
  python run.py --check                    # 检查环境、数据、API Key，并测试一次 API 调用
  python run.py --dry-run                  # 只渲染文件、拼提示词，不调用 API
  python run.py --limit 20                 # 先跑前 20 题试试
  python run.py --ids 1-50,63              # 跑指定题号
  python run.py --file 006.pdf             # 跑某个文件的所有题
  python run.py --types structure          # 只跑某类题
  python run.py                            # 跑全部题，并生成提交文件
  python run.py --submit-only              # 不调用 API，用已有结果生成提交文件
  python run.py --list-results             # 查看各模型已有多少题的结果
  python run.py --tag qwen3.7-plus_think --thinking --refinalize
                                           # 改了清洗规则后，不调用 API 重新生成答案和提交文件
  python run.py --models qwen3.8-max,qwen3.6-plus --thinking --check
                                           # 逐个测试候选模型能不能用（免费额度是否还有）
  python run.py --model qwen3.7-plus --thinking --submit-only --fallback qwen3-vl-plus
                                           # 新模型的答案为主，缺的题用 qwen3-vl-plus 的补
  python run.py --show 32                  # 查看某题的提示词、模型原始输出和最终答案
  python run.py --votes 3                  # 非结构题采样 3 次投票（更稳，费用约 3 倍）
"""
import argparse
import json
import sys
from pathlib import Path

from tablebot.config import Config
from tablebot.data import load_questions, parse_id_spec
from tablebot.pipeline import load_results, model_tag, refinalize, results_dir, run, write_submission


def build_cfg(a) -> Config:
    cfg = Config()
    if a.data:
        cfg.data_dir = Path(a.data)
    if a.model:
        cfg.model = a.model
    if a.models:
        cfg.models = [m.strip() for m in a.models.split(",") if m.strip()]
    if cfg.models:
        cfg.model = cfg.models[0]  # 结果目录按第一个模型命名
    if a.thinking:
        cfg.enable_thinking = True
    if a.workers:
        cfg.workers = a.workers
    if a.tag:
        cfg.result_tag = a.tag
    if a.votes:
        cfg.votes = a.votes
    return cfg


def select(questions, a, cfg):
    qs = questions
    if a.ids:
        want = set(parse_id_spec(a.ids))
        qs = [q for q in qs if q["id"] in want]
    if a.file:
        qs = [q for q in qs if q["file_name"] == a.file or Path(q["path"]).name == a.file]
    if a.types:
        ts = set(a.types.split(","))
        qs = [q for q in qs if q["question_type"] in ts]
    if a.only_missing:
        res = load_results(cfg)
        qs = [q for q in qs if q["id"] not in res or res[q["id"]].get("error") or not res[q["id"]].get("answer")]
    if a.limit:
        qs = qs[: a.limit]
    return qs


def check(cfg: Config, questions):
    print(f"数据目录：{cfg.data_dir}")
    print(f"题目数：{len(questions)}")
    fixes = [q for q in questions if q["fixes"]]
    for q in fixes:
        print(f"  已自动修正 id={q['id']}: {'; '.join(q['fixes'])}")
    print(f"模型：{', '.join(cfg.models or [cfg.model])}   思考模式：{cfg.enable_thinking}   接口：{cfg.base_url}")
    if not cfg.api_key:
        print("!! 没有找到 DASHSCOPE_API_KEY，请在 .env 里填写")
        return
    print(f"API Key：{cfg.api_key[:5]}...{cfg.api_key[-4:]}")
    from tablebot.llm import QwenClient
    import copy
    for m in (cfg.models or [cfg.model]):
        c = copy.copy(cfg)
        c.model, c.models = m, [m]
        try:
            r = QwenClient(c, use_cache=False).chat(
                [{"role": "user", "content": '只输出这个 JSON：{"ok": true}'}])
            print(f"  {m:<24} 可用（{r['seconds']}s）")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            why = ("免费额度已用完" if "FreeTierOnly" in msg or "AllocationQuota" in msg else
                   "账户欠费" if "Arrearage" in msg else "模型不存在或未开通" if "not" in msg.lower() else msg[:100])
            print(f"  {m:<24} 不可用：{why}")


def show(cfg: Config, questions, qid: int):
    from tablebot.prompts import build_messages
    from tablebot.render import load_document
    q = next((q for q in questions if q["id"] == qid), None)
    if not q:
        print("没有这道题")
        return
    print(json.dumps({k: v for k, v in q.items()}, ensure_ascii=False, indent=1))
    doc = load_document(q["path"], cfg)
    msgs = build_messages(q, doc)
    print("\n======== 提示词（图片省略） ========")
    for c in msgs[1]["content"]:
        print("<图片>" if c["type"] == "image_url" else c["text"])
    r = load_results(cfg).get(qid)
    if r:
        print("\n======== 模型原始输出 ========")
        print(r.get("raw", ""))
        print("\n======== 最终答案 ========")
        print(r.get("answer"), "  错误:" if r.get("error") else "", r.get("error") or "")
        if r.get("notes"):
            print("备注：", r["notes"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="复杂表格识别 baseline")
    p.add_argument("--data", help="数据目录（含 files/ 和 tests.xlsx）")
    p.add_argument("--model", help="模型名，如 qwen3-vl-plus / qwen-vl-max")
    p.add_argument("--models", help="多模型轮换，逗号分隔，如 qwen3.8-max,qwen3.6-plus；前一个免费额度用完自动换下一个")
    p.add_argument("--thinking", action="store_true", help="开启思考模式")
    p.add_argument("--tag", help="结果目录名（见 --list-results），用来接着往某个已有结果里补题")
    p.add_argument("--workers", type=int, help="并发数")
    p.add_argument("--votes", type=int, help="非结构题采样几次投票，如 3（费用相应增加）")
    p.add_argument("--ids", help="题号，如 1-20,63")
    p.add_argument("--file", help="只跑某个文件，如 006.pdf")
    p.add_argument("--types", help="题型，逗号分隔：structure,extract,thinking")
    p.add_argument("--limit", type=int, help="最多跑几题")
    p.add_argument("--only-missing", action="store_true", help="只跑还没有成功结果的题")
    p.add_argument("--no-cache", action="store_true", help="忽略缓存重新请求")
    p.add_argument("--dry-run", action="store_true", help="不调用 API")
    p.add_argument("--no-submit", action="store_true", help="跑完不生成提交文件")
    p.add_argument("--submit-only", action="store_true", help="只用已有结果生成提交文件")
    p.add_argument("--check", action="store_true", help="检查环境和 API")
    p.add_argument("--fallback", help="生成提交文件时，当前模型没有答案的题用这些结果补，如 qwen3-vl-plus（逗号分隔，见 --list-results）")
    p.add_argument("--list-results", action="store_true", help="列出各模型已有的结果数量")
    p.add_argument("--refinalize", action="store_true", help="不调用 API，用最新的清洗规则重新生成答案，然后生成提交文件")
    p.add_argument("--show", type=int, help="查看某题详情")
    a = p.parse_args()

    cfg = build_cfg(a)
    if not cfg.tests_path.exists():
        sys.exit(f"找不到 {cfg.tests_path}，请用 --data 指定数据目录")
    questions = load_questions(cfg.tests_path, cfg.files_dir)

    if a.check:
        return check(cfg, questions)
    if a.list_results:
        import json as _j
        base = results_dir(cfg).parent
        for d in sorted(x for x in base.iterdir() if x.is_dir()):
            rs = [_j.loads(f.read_text(encoding="utf-8")) for f in d.glob("*.json")]
            ok = sum(1 for r in rs if r.get("answer") and not r.get("error"))
            print(f"{d.name:<28} 成功 {ok:>4} 题，失败 {len(rs) - ok:>4} 题")
        return
    if a.show:
        return show(cfg, questions, a.show)
    fallbacks = [x.strip() for x in a.fallback.split(",")] if a.fallback else None
    if a.refinalize:
        refinalize(questions, cfg)
        write_submission(questions, cfg, fallbacks=fallbacks)
        return
    if a.submit_only:
        write_submission(questions, cfg, fallbacks=fallbacks)
        return

    qs = select(questions, a, cfg)
    print(f"本次处理 {len(qs)} 题（模型 {", ".join(cfg.models or [cfg.model])}，思考模式 {cfg.enable_thinking}，并发 {cfg.workers}，投票 {cfg.votes}）")
    if not qs:
        return
    recs = run(qs, cfg, use_cache=not a.no_cache, dry_run=a.dry_run)
    if a.dry_run:
        pages = sum(r.get("pages", 0) for r in recs)
        errs = [r for r in recs if r["error"]]
        est = sum(r.get("est_tokens", 0) for r in recs)
        print(f"dry-run：共 {pages} 张页面图片；出错 {len(errs)} 题；预计输入约 {est / 1e6:.2f}M tokens"
              f"（每题平均 {est / max(1, len(recs)):.0f}），输出约 {len(recs) * 300 / 1e6:.2f}M tokens（不含思考）")
        for r in errs[:20]:
            print("  ", r["id"], r["error"])
        return
    if not a.no_submit:
        write_submission(questions, cfg, fallbacks=fallbacks)


if __name__ == "__main__":
    main()
