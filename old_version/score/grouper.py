def group_by_score(results: list[dict]) -> list[dict]:
    valid      = [r for r in results if r.get("score_a") is not None and r.get("score_b") is not None]
    unreadable = [r for r in results if r.get("score_a") is None     or  r.get("score_b") is None]

    score_map: dict[tuple[int, int], list[str]] = {}
    for r in valid:
        key = (int(r["score_a"]), int(r["score_b"]))
        score_map.setdefault(key, []).append(r["clip"])

    sorted_scores = sorted(score_map.items(), key=lambda x: (x[0][0] + x[0][1], x[0][0]))

    groups: list[dict] = []
    prev: tuple[int, int] | None = None

    for (a, b), clips in sorted_scores:
        needs_review = False
        reason       = ""

        if prev is not None:
            pa, pb = prev
            da, db = a - pa, b - pb
            if da < 0 or db < 0:
                needs_review = True
                reason = f"分數倒退: ({pa},{pb}) → ({a},{b})"
            elif da > 0 and db > 0:
                needs_review = True
                reason = f"雙方同時加分 (不合理): ({pa},{pb}) → ({a},{b})"
            elif da + db > 1:
                needs_review = True
                reason = f"跳 {da + db} 分 (可能有缺漏片段): ({pa},{pb}) → ({a},{b})"

        groups.append({
            "score_a":       a,
            "score_b":       b,
            "clips":         sorted(clips),
            "needs_review":  needs_review,
            "review_reason": reason,
        })
        prev = (a, b)

    if unreadable:
        groups.append({
            "score_a":       None,
            "score_b":       None,
            "clips":         sorted(r["clip"] for r in unreadable),
            "needs_review":  True,
            "review_reason": "無法辨識分數",
        })

    return groups
