#!/usr/bin/env python3
"""AgentBrain: the central loop — observe state, call a tool, repeat, then answer.

One JSON decision per round; invalid/unknown tool calls consume budget but return an
error observation so the brain can recover. Generation + answer-utility run once at
the end over the accumulated evidence set.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Optional

from backends.rag_backend import RagPipelineBackend
from backends.llm_backend import BrainBackend
from backends.reward_backend import RewardBackend
from prompts.brain_prompt import build_system_prompt, render_state_message
from runtime.session import AgentSession
from schemas import Observation, Round, ToolCall, Trajectory
from tools import ToolRegistry, ToolResult, default_registry


class AgentBrain:
    def __init__(
        self,
        config: Dict[str, Any],
        rag: RagPipelineBackend,
        brain: BrainBackend,
        reward: RewardBackend,
        registry: Optional[ToolRegistry] = None,
    ):
        self.config = config
        self._rag = rag
        self.brain = brain
        self.reward = reward
        self.registry = registry or default_registry()
        self.system_prompt = build_system_prompt(self.registry.specs())

    # ------------------------------------------------------------------ episode
    def solve(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run one question end-to-end; returns a v0.5-compatible trajectory dict."""
        gold_source = dict(item.get("gold_source") or item.get("source") or {})
        session = AgentSession(
            config=self.config,
            rag=self._rag,
            qid=str(item.get("qid", "")),
            question=str(item.get("question", item.get("query_text", ""))),
            options=dict(item.get("options", {}) or {}),
            query_image_path=str(item.get("query_image_path", "") or ""),
            retrieval_hint=str(item.get("retrieval_hint_zh", "") or ""),
            # v0.3 P0 去泄露：题目为库内来源时排除自身样本与同书 doc_id
            exclude_sample_ids=(str(gold_source.get("sample_id") or ""),),
            exclude_doc_ids=(str(gold_source.get("doc_id") or ""),),
        )
        rounds: List[Round] = []
        finish_reason = ""

        while not session.budget_exhausted() and not session.submitted:
            state_msg = render_state_message(
                last_tool=rounds[-1].call.tool if rounds else "",
                last_message=rounds[-1].observation.message if rounds else "",
                passages=session.last_passages,
                collected=session.collected,
                search_history=session.search_history,
                rounds_used=session.rounds_used,
                tool_calls_used=session.tool_calls_used,
                max_rounds=session.max_rounds,
                max_tool_calls=session.max_tool_calls,
                retrieval_hint=session.retrieval_hint,
                final_round=(session.rounds_used + 1) >= session.max_rounds,
            )
            call, raw_text = self._decide(state_msg, session.query_image_path)
            result = self._execute(session, call)
            # every decision consumes budget, so failed/invalid calls cannot loop forever
            session.advance()
            obs = Observation(
                tool=call.tool,
                ok=result.ok,
                candidates=result.candidates,
                n_new=result.n_new,
                message=result.message,
                state=session.state_summary(),
            )
            rounds.append(Round(index=len(rounds), call=call, observation=obs))
            if result.finish:
                finish_reason = "submitted"
                break

        if not finish_reason:
            finish_reason = "budget_exhausted"

        # ---- generator answers from the accumulated evidence set ----
        gold = str(item.get("answer", "") or "")
        error: Optional[Dict[str, str]] = None
        try:
            gen = self.reward.generate(
                session.question, session.options, session.collected, image_path=session.query_image_path
            )
            prediction = gen.get("prediction", "")
            response = gen.get("response", "")
            u_set = self.reward.utility_from_generation(gen, gold)
        except Exception:
            prediction, response, u_set = "", "", None
            error = {"stage": "generation", "detail": traceback.format_exc(limit=3)}
        correct = (prediction == gold) if (gold and prediction) else None

        traj = Trajectory(
            qid=session.qid,
            question=session.question,
            options=session.options,
            answer=gold,
            answer_text=str(item.get("answer_text", "") or ""),
            query_image_path=session.query_image_path,
            original_query=str(item.get("original_query", "") or ""),
            gold_source=dict(item.get("gold_source", {}) or {}),
            rounds=rounds,
            final_evidence=session.collected,
            generator_response=response,
            prediction=prediction,
            correct=correct,
            u_set=u_set,
            finish_reason=finish_reason,
            n_rounds=session.rounds_used,
            n_tool_calls=session.tool_calls_used,
            error=error,
        )
        return traj.compat_dict()

    # ------------------------------------------------------------------ internals
    def _decide(self, state_msg: str, image_path: str = ""):
        try:
            decision = self.brain.decide(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": state_msg},
                ],
                image_path=image_path,
            )
            raw_text = json.dumps(decision, ensure_ascii=False)
            call = ToolCall(
                thought=str(decision.get("thought", "")),
                tool=str(decision.get("tool", "")),
                args=dict(decision.get("args") or {}),
                raw_text=raw_text,
            )
        except Exception as e:
            call = ToolCall(
                thought="brain call failed",
                tool="_invalid",
                args={},
                raw_text="",
            )
            call.args = {"_error": str(e)}
        return call, call.raw_text

    def _execute(self, session: AgentSession, call: ToolCall) -> ToolResult:
        try:
            tool = self.registry.get(call.tool)
        except KeyError:
            return ToolResult(ok=False, message=f"unknown tool {call.tool!r}; use one of {self.registry.names()}")
        try:
            return tool.run(session, call.args)
        except Exception as e:
            return ToolResult(ok=False, message=f"tool execution error: {e}")
