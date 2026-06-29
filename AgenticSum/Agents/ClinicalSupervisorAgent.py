from typing import Dict, Optional, Any, List


class ClinicalSupervisorAgent:
    """
    ClinicalSupervisorAgent governs the full agentic summarization pipeline.
    It orchestrates input compression, draft generation, hallucination detection,
    targeted correction, and convergence-based termination.

    This agent performs NO generation and calls NO language models directly.
    """

    def __init__(
        self,
        focus_agent,
        draft_agent,
        hallucination_detector_agent,
        fix_agent,
        max_iterations: int = 3,
    ):
        self.focus_agent = focus_agent
        self.draft_agent = draft_agent
        self.hallucination_detector_agent = hallucination_detector_agent
        self.fix_agent = fix_agent

        self.max_iterations = max_iterations
        self.iteration = 0
        self.previous_hallucination_mask: Optional[Dict[int, int]] = None

    # ======================================================
    # Public API (single entry point)
    # ======================================================
    def run(self, document: str) -> Dict[str, Any]:
        """
        Executes the full agentic summarization pipeline and
        returns structured outputs for analysis and evaluation.
        """

        # -------- RESET STATE --------
        self.reset()
        print("[Supervisor] Starting new document")

        # --------------------------------------------------
        # Step 1: Input compression
        # --------------------------------------------------
        focus_outputs = self.focus_agent.compress(document)
        reduced_sentences: List[str] = focus_outputs["sentences"]
        print(f"[Supervisor] FocusAgent retained {len(reduced_sentences)} sentences")

        # --------------------------------------------------
        # Step 2: Initial draft
        # --------------------------------------------------
        draft_summary = self.draft_agent.generate(reduced_sentences)
        print("[Supervisor] DraftAgent generated initial summary")

        summary = draft_summary  # working copy
        final_detection_outputs: Optional[Dict[str, Any]] = None
        final_hallucinated_spans: List[str] = []
        final_decision: Optional[Dict[str, Any]] = None

        # --------------------------------------------------
        # Step 3: Iterative verification + repair
        # --------------------------------------------------
        while True:
            detection_outputs: Dict[str, Any] = (
                self.hallucination_detector_agent.analyze(
                    source_document=document,
                    draft_summary=summary,
                )
            )

            final_detection_outputs = detection_outputs

            hallucination_mask: Dict[int, int] = detection_outputs["hallucination_mask"]
            spans: List[str] = detection_outputs["spans"]

            hallucinated_spans = [
                spans[j] for j, v in hallucination_mask.items() if v == 1
            ]
            final_hallucinated_spans = hallucinated_spans

            decision = self.check_convergence(hallucination_mask)
            final_decision = decision

            print(
                f"[Supervisor] Iter {self.iteration} | "
                f"hallucinated spans: {len(hallucinated_spans)} | "
                f"decision: {decision['reason']}"
            )

            if decision["terminate"]:
                break

            # --------------------------------------------------
            # Step 4: Targeted repair
            # --------------------------------------------------
            self.iteration += 1

            fixed_summary = self.fix_agent.fix(
                source_document=document,
                spans=spans,
                hallucination_mask=hallucination_mask,
            )

            summary = fixed_summary  # update working summary

        print("[Supervisor] Pipeline finished")

        # --------------------------------------------------
        # Step 5: Structured return (✔ CLEAN + EXPLICIT)
        # --------------------------------------------------
        return {
            "draft_summary": draft_summary,
            "fixed_summary": summary,
            "hallucinated_spans": final_hallucinated_spans,
            "hallucination_mask": hallucination_mask,
            "spans": spans,
            "num_iterations": self.iteration,
            "termination_reason": final_decision["reason"]
            if final_decision is not None
            else None,
        }

    # ======================================================
    # Convergence logic
    # ======================================================
    def check_convergence(
        self,
        hallucination_mask: Dict[int, int],
    ) -> Dict[str, Any]:

        num_hallucinations = sum(hallucination_mask.values())

        if num_hallucinations == 0:
            return {"terminate": True, "reason": "no_hallucinations_remaining"}

        if self.iteration >= self.max_iterations:
            return {"terminate": True, "reason": "max_iterations_reached"}

        if (
            self.previous_hallucination_mask is not None
            and hallucination_mask == self.previous_hallucination_mask
        ):
            return {"terminate": True, "reason": "hallucination_pattern_stabilized"}

        self.previous_hallucination_mask = hallucination_mask.copy()
        return {"terminate": False, "reason": "hallucinations_detected"}

    # ======================================================
    # Utility
    # ======================================================
    def reset(self):
        self.iteration = 0
        self.previous_hallucination_mask = None
        
        # Reset detector state (clears cached spans)
        self.hallucination_detector_agent.reset()
        
        # Reset semantic judge cache
        if hasattr(self.hallucination_detector_agent, 'semantic_judge'):
            self.hallucination_detector_agent.semantic_judge.reset()