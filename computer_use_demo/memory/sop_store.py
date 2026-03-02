"""
SOP 存储与召回，参考 pc-agent-loop 的 L3 任务级记录。
"""
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from computer_use_demo.tools.logger import logger


class TaskSOP:
    """任务 SOP 数据结构。steps 为可读的语义描述字符串（新格式）或旧格式的 list[dict]。"""

    def __init__(
        self,
        task_description: str,
        steps: str | list[dict],
        *,
        sop_id: str | None = None,
        keywords: list[str] | None = None,
        success_count: int = 1,
        created_at: str | None = None,
        notes: str = "",
    ):
        self.sop_id = sop_id or f"sop_{uuid.uuid4().hex[:12]}"
        self.task_description = task_description
        self.keywords = keywords or self._extract_keywords(task_description)
        self.steps = steps  # str: 可读语义描述；list: 旧格式兼容
        self.success_count = success_count
        self.created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M")
        self.notes = notes

    def _extract_keywords(self, text: str) -> list[str]:
        """从任务描述中提取关键词。中文用 2-gram，英文用单词。"""
        words: set[str] = set()
        # 英文/数字 2+ 字符
        for w in re.findall(r"[a-zA-Z0-9]{2,}", text.lower()):
            words.add(w)
        # 中文：2 字符滑动窗口，避免整句被当作一个词
        for i in range(len(text) - 1):
            c1, c2 = text[i], text[i + 1]
            if "\u4e00" <= c1 <= "\u9fff" and "\u4e00" <= c2 <= "\u9fff":
                words.add(c1 + c2)
        return list(words)[:15]

    def to_dict(self) -> dict:
        return {
            "sop_id": self.sop_id,
            "task_description": self.task_description,
            "keywords": self.keywords,
            "steps": self.steps,
            "success_count": self.success_count,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSOP":
        steps = d.get("steps", "")
        # 兼容：steps 为 str 直接使用；为 list[dict] 保留旧格式；为 list[str] 转成字符串
        if isinstance(steps, list) and steps:
            if isinstance(steps[0], dict):
                pass  # 旧格式 list[dict]
            else:
                steps = "\n".join(str(s) for s in steps)
        elif not isinstance(steps, str):
            steps = ""
        return cls(
            task_description=d["task_description"],
            steps=steps,
            sop_id=d.get("sop_id"),
            keywords=d.get("keywords"),
            success_count=d.get("success_count", 1),
            created_at=d.get("created_at"),
            notes=d.get("notes", ""),
        )

    def to_prompt_hint(self, max_steps: int = 10) -> str:
        """将 SOP 转为可注入 Planner 的提示文本。"""
        if isinstance(self.steps, str) and self.steps.strip():
            return (
                f"\n[RECALLED SOP] 曾成功完成类似任务「{self.task_description}」，参考步骤：\n"
                + self.steps.strip()
                + "\n可根据当前屏幕灵活调整，不必完全照搬。\n"
            )
        # 旧格式兼容
        steps_str = []
        for i, s in enumerate((self.steps or [])[:max_steps], 1):
            if isinstance(s, dict):
                desc = s.get("description", "").strip()
                if desc:
                    steps_str.append(f"  {i}. {desc}")
                else:
                    a = s.get("action", "")
                    v = s.get("value")
                    p = s.get("position")
                    part = f"  {i}. {a}"
                    if v:
                        part += f" value={repr(v)[:50]}"
                    if p:
                        part += f" position={p}"
                    steps_str.append(part)
        return (
            f"\n[RECALLED SOP] 曾成功完成类似任务「{self.task_description}」，参考步骤：\n"
            + "\n".join(steps_str)
            + "\n可根据当前屏幕灵活调整，不必完全照搬。\n"
        )


class SOPStore:
    """SOP 持久化存储"""

    def __init__(self, base_dir: str | Path = "./memory"):
        self.base_dir = Path(base_dir)
        self.sops_dir = self.base_dir / "sops"
        self.sops_dir.mkdir(parents=True, exist_ok=True)

    def _sop_path(self, sop_id: str) -> Path:
        return self.sops_dir / f"{sop_id}.json"

    def save(self, sop: TaskSOP) -> Path:
        """保存 SOP"""
        path = self._sop_path(sop.sop_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sop.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"SOP 已保存: {path}")
        return path

    def load(self, sop_id: str) -> TaskSOP | None:
        """加载单个 SOP"""
        path = self._sop_path(sop_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return TaskSOP.from_dict(json.load(f))

    def list_all(self) -> list[TaskSOP]:
        """列出所有 SOP"""
        sops = []
        for p in self.sops_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    sops.append(TaskSOP.from_dict(json.load(f)))
            except Exception as e:
                logger.warning(f"加载 SOP 失败 {p}: {e}")
        return sorted(sops, key=lambda s: s.created_at, reverse=True)

    def _task_to_words(self, text: str) -> set[str]:
        """将任务文本转为可匹配的词集合（与 TaskSOP._extract_keywords 一致）"""
        words: set[str] = set()
        for w in re.findall(r"[a-zA-Z0-9]{2,}", text.lower()):
            words.add(w)
        for i in range(len(text) - 1):
            c1, c2 = text[i], text[i + 1]
            if "\u4e00" <= c1 <= "\u9fff" and "\u4e00" <= c2 <= "\u9fff":
                words.add(c1 + c2)
        return words

    def recall(self, user_task: str, top_k: int = 3) -> list[TaskSOP]:
        """
        根据用户任务召回相关 SOP。
        关键词匹配：用户任务与 SOP 的 keywords 交集越多，得分越高。
        """
        task_words = self._task_to_words(user_task)

        scored: list[tuple[float, TaskSOP]] = []
        for sop in self.list_all():
            sop_words = set(sop.keywords) | self._task_to_words(sop.task_description)
            overlap = len(task_words & sop_words)
            if overlap > 0:
                # 考虑成功次数作为权重
                score = overlap * (1 + 0.1 * min(sop.success_count, 10))
                scored.append((score, sop))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top_k]]

    def delete(self, sop_id: str) -> bool:
        """删除 SOP"""
        path = self._sop_path(sop_id)
        if path.exists():
            path.unlink()
            return True
        return False
