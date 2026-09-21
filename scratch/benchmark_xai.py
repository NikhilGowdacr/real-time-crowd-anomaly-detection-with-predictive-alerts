"""
Micro-benchmark measuring latency of ExplanationEngine across 5,000 iterations.
"""
import time
from src.decision.decision_record import DecisionEvent
from src.xai.explanation_engine import ExplanationEngine

def benchmark():
    engine = ExplanationEngine()
    event = DecisionEvent(
        timestamp=time.time(),
        previous_state="SUSPICIOUS",
        current_state="HIGH_RISK",
        score=0.74,
        confidence=0.82,
        reason="Physical altercation observed with rapid dispersion.",
        dominant_modality="bimodal",
        video_score=0.75,
        audio_score=0.70,
        weapon_score=0.0,
        fight_score=0.78,
        corroborated=True,
        critical_evidence=False,
        evidence_summary={
            "synergy_type": "FIGHT_AND_SHOUTING",
            "video": {
                "fight_score": 0.78,
                "movement_score": 0.65,
                "dispersion_rate": 8.5,
                "direction_entropy": 2.2,
                "density_m2": 2.1,
                "density_level": "HIGH",
            },
            "audio": {"audio_event": "shouting", "audio_confidence": 0.75, "is_active": True},
        },
    )

    # Warmup
    for _ in range(200):
        engine.explain_decision(event)

    iterations = 5000
    t0 = time.perf_counter()
    for _ in range(iterations):
        engine.explain_decision(event)
    t1 = time.perf_counter()

    total_time_ms = (t1 - t0) * 1000.0
    avg_latency_ms = total_time_ms / iterations

    print(f"XAI Benchmark Results:")
    print(f"Iterations   : {iterations}")
    print(f"Total Time   : {total_time_ms:.2f} ms")
    print(f"Avg Latency  : {avg_latency_ms:.4f} ms per explanation")
    print(f"Target (<2ms): {'PASSED' if avg_latency_ms < 2.0 else 'FAILED'}")

if __name__ == "__main__":
    benchmark()
