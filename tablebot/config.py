"""全局配置。命令行参数会覆盖这里的默认值。"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录 = run.py 所在目录
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """读取项目根目录下的 .env（KEY=VALUE 每行一个），不覆盖已有环境变量。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass
class Config:
    # ---- 路径 ----
    data_dir: Path = ROOT / "multimodal_table_recognition"  # 内含 files/ 与 tests.xlsx
    work_dir: Path = ROOT / "work"        # 缓存、中间结果
    out_dir: Path = ROOT / "outputs"      # 提交文件

    # ---- 模型（阿里云百炼 OpenAI 兼容接口）----
    base_url: str = os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key: str = os.environ.get("DASHSCOPE_API_KEY", "")
    model: str = os.environ.get("QWEN_MODEL", "qwen3-vl-plus")
    # 多模型轮换（用免费额度）：按顺序使用，前一个免费额度用完自动换下一个。为空则只用 model
    models: list = field(default_factory=lambda: [m.strip() for m in os.environ.get("QWEN_MODELS", "").split(",") if m.strip()])
    enable_thinking: bool = False         # 打开后更准但更慢更贵
    result_tag: str = ""                  # 结果目录名；为空时按 第一个模型名(+_think) 命名
    temperature: float = 0.0
    max_tokens: int = 16384
    timeout: int = 300                    # 单次请求超时（秒）
    max_retries: int = 5
    workers: int = 4                      # 并发请求数，遇到限流就调小
    votes: int = 1                        # >1 时非结构题多次采样投票（费用 × votes）
    vote_temperature: float = 0.7

    # ---- 图像 ----
    pdf_scale: float = 2.5                # PDF 渲染倍率（72dpi * 2.5 = 180dpi）
    max_side: int = 2400                  # 图片最长边上限
    min_side: int = 1000                  # 小图放大到最长边至少这么多
    scan_scale: float = 3.5               # 扫描件（无文字层的 PDF）用更高分辨率，字太小时容易看错数字
    scan_max_side: int = 3200
    pdf_text_chars: int = 12000           # 附带的 PDF 文字层最大字符数

    # ---- 答案格式 ----
    array_numeric_as_number: bool = True  # json_array 中纯数字字符串转成数字
    trust_expression: bool = True         # 推理题用程序计算的算式结果覆盖模型的心算结果

    extra: dict = field(default_factory=dict)

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def tests_path(self) -> Path:
        return self.data_dir / "tests.xlsx"
