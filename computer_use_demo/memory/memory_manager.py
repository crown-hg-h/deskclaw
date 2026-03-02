"""
记忆系统入口，整合 L0/L2/L3。
参考 pc-agent-loop: https://github.com/lsdefine/pc-agent-loop
"""
from pathlib import Path

from computer_use_demo.tools.logger import logger

from .sop_store import SOPStore, TaskSOP


class MemoryManager:
    """
    记忆管理器
    - L2: global_facts - 环境事实（路径、配置等）
    - L3: SOP 存储 - 任务级操作流程
    """

    def __init__(self, base_dir: str | Path = "./memory"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sop_store = SOPStore(self.base_dir)
        self.global_facts_path = self.base_dir / "global_facts.json"
        self._global_facts: dict = {}
        self._load_global_facts()

    def _load_global_facts(self) -> None:
        if self.global_facts_path.exists():
            try:
                import json
                with open(self.global_facts_path, "r", encoding="utf-8") as f:
                    self._global_facts = json.load(f)
            except Exception as e:
                logger.warning(f"加载 global_facts 失败: {e}")

    def get_global_facts(self) -> dict:
        return self._global_facts.copy()

    def update_global_fact(self, key: str, value: str | dict) -> None:
        self._global_facts[key] = value
        import json
        with open(self.global_facts_path, "w", encoding="utf-8") as f:
            json.dump(self._global_facts, f, ensure_ascii=False, indent=2)

    def recall_sops(self, user_task: str, top_k: int = 3) -> list[TaskSOP]:
        """召回与用户任务相关的 SOP"""
        return self.sop_store.recall(user_task, top_k=top_k)

    def recall_sops_as_prompt(self, user_task: str, top_k: int = 2) -> str:
        """召回 SOP 并转为可注入 system prompt 的文本"""
        sops = self.recall_sops(user_task, top_k=top_k)
        if not sops:
            return ""
        return "\n".join(s.to_prompt_hint() for s in sops)

    def save_sop(
        self,
        task_description: str,
        steps: str | list[dict],
        *,
        keywords: list[str] | None = None,
        notes: str = "",
    ) -> TaskSOP:
        """保存新 SOP 或更新已有 SOP。steps 为可读的语义描述字符串或旧格式 list[dict]。"""
        # 检查是否有相似 SOP，若有则增加 success_count
        recalled = self.recall_sops(task_description, top_k=1)
        if recalled:
            best = recalled[0]
            # 简单判断：描述相似则视为同一类任务，增加计数
            if best.task_description == task_description or (
                len(set(task_description.split()) & set(best.task_description.split())) >= 2
            ):
                best.success_count += 1
                best.steps = steps
                self.sop_store.save(best)
                return best

        sop = TaskSOP(
            task_description=task_description,
            steps=steps,
            keywords=keywords,
            notes=notes,
        )
        self.sop_store.save(sop)
        return sop

    def list_sops(self) -> list[TaskSOP]:
        return self.sop_store.list_all()

    def delete_sop(self, sop_id: str) -> bool:
        return self.sop_store.delete(sop_id)
