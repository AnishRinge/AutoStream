"""Build internal knowledge records for semantic retrieval.

Records are derived from verified JSON facts only (no new business claims).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    text: str
    category: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def build_knowledge_records(kb: dict[str, Any], *, source: str = "knowledge_base/data.json") -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    pricing = kb.get("pricing") or {}
    policies = kb.get("policies") or {}

    for plan_id in ("basic", "pro"):
        plan = pricing.get(plan_id) or {}
        price = plan.get("price")
        features = plan.get("features") or []
        if price and isinstance(features, list):
            title = "Pro plan" if plan_id == "pro" else "Basic plan"
            feature_text = "; ".join(str(f) for f in features)
            text = (
                f"AutoStream {title} pricing and features. "
                f"Price: {price}. Features: {feature_text}."
            )
            records.append(
                KnowledgeRecord(
                    id=f"pricing.{plan_id}",
                    text=text,
                    category="pricing",
                    source=source,
                    metadata={"plan": plan_id, "price": price, "features": list(features)},
                )
            )
            for idx, feat in enumerate(features):
                feat_text = (
                    f"AutoStream {title} feature: {feat}. Plan price: {price}."
                )
                records.append(
                    KnowledgeRecord(
                        id=f"pricing.{plan_id}.feature.{idx}",
                        text=feat_text,
                        category="pricing_feature",
                        source=source,
                        metadata={"plan": plan_id, "price": price, "feature": feat},
                    )
                )

    refund = policies.get("refund")
    if isinstance(refund, str) and refund.strip():
        records.append(
            KnowledgeRecord(
                id="policy.refund",
                text=f"AutoStream refund policy: {refund}",
                category="policy",
                source=source,
                metadata={"policy": "refund", "value": refund},
            )
        )

    support = policies.get("support")
    if isinstance(support, str) and support.strip():
        records.append(
            KnowledgeRecord(
                id="policy.support",
                text=f"AutoStream support policy and contact: {support}",
                category="policy",
                source=source,
                metadata={"policy": "support", "value": support},
            )
        )

    return records


def records_by_id(records: Iterable[KnowledgeRecord]) -> dict[str, KnowledgeRecord]:
    return {r.id: r for r in records}
