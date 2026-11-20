import pytest
from services.cache_service import CacheService


def test_exact_cache_hit_and_miss(test_cache_service):
    q = "What is the company meal expense allowance?"
    answer = "The daily meal expense allowance is $75 with itemized receipts."
    allowed_depts = ["ALL", "FIN"]

    # 1. Initially must be a MISS
    cached_ans, meta, hit_type = test_cache_service.get_cached_answer(
        question=q,
        allowed_departments=allowed_depts,
    )
    assert cached_ans is None
    assert hit_type == "MISS"

    # 2. Store in cache
    test_cache_service.set_cached_answer(
        question=q,
        answer=answer,
        allowed_departments=allowed_depts,
        metadata={"citations": [{"filename": "Expense_Policy.pdf"}]},
        tokens_used=420,
    )

    # 3. Subsequent exact query should be a HIT
    cached_ans2, meta2, hit_type2 = test_cache_service.get_cached_answer(
        question=q,
        allowed_departments=allowed_depts,
    )
    assert hit_type2 == "EXACT"
    assert cached_ans2 == answer
    assert meta2["citations"][0]["filename"] == "Expense_Policy.pdf"


def test_cache_department_acl_isolation(test_cache_service):
    """
    CRITICAL SECURITY TEST:
    A user with only ['ALL', 'IT'] permissions must NEVER retrieve a cached answer
    containing HR or Finance information cached for ['ALL', 'HR', 'FIN'].
    """
    q = "What is the bonus compensation formula?"
    secret_hr_answer = "Executive bonus pool is 15% of annual operating profit."
    hr_depts = ["ALL", "HR"]
    it_depts = ["ALL", "IT"]

    # Store for HR
    test_cache_service.set_cached_answer(
        question=q,
        answer=secret_hr_answer,
        allowed_departments=hr_depts,
    )

    # IT user asks the exact same question
    cached_ans, _, hit_type = test_cache_service.get_cached_answer(
        question=q,
        allowed_departments=it_depts,
    )

    # IT user MUST get a MISS and NOT see the HR cached answer
    assert cached_ans is None
    assert hit_type == "MISS"


def test_semantic_cache_hit(test_cache_service):
    """Verifies that semantically similar vectors trigger a semantic cache hit."""
    q_original = "How many vacation days do full-time employees get?"
    ans = "Full-time employees receive 20 paid vacation days annually."
    allowed_depts = ["ALL", "HR"]

    # Unit vector representing original query
    emb_original = [0.1] * 1536
    norm = sum(x**2 for x in emb_original) ** 0.5
    emb_original = [x / norm for x in emb_original]

    test_cache_service.set_cached_answer(
        question=q_original,
        answer=ans,
        allowed_departments=allowed_depts,
        query_embedding=emb_original,
    )

    # Similar query vector (cosine similarity ~0.999)
    emb_similar = list(emb_original)
    emb_similar[0] += 0.001
    norm_sim = sum(x**2 for x in emb_similar) ** 0.5
    emb_similar = [x / norm_sim for x in emb_similar]

    cached_ans, meta, hit_type = test_cache_service.get_cached_answer(
        question="What is the annual PTO allowance for employees?",
        allowed_departments=allowed_depts,
        query_embedding=emb_similar,
    )

    assert hit_type == "SEMANTIC"
    assert cached_ans == ans
    assert meta["semantic_score"] >= test_cache_service.semantic_threshold


def test_cache_invalidation_on_document_update(test_cache_service):
    """Cache invalidation must increment KB version, making old cached entries obsolete."""
    q = "What is the maternity leave policy?"
    ans = "12 weeks of paid leave."
    allowed_depts = ["ALL", "HR"]

    test_cache_service.set_cached_answer(
        question=q,
        answer=ans,
        allowed_departments=allowed_depts,
    )

    # Confirms it is cached
    cached1, _, hit1 = test_cache_service.get_cached_answer(q, allowed_depts)
    assert hit1 == "EXACT"

    # Invalidate cache for HR department
    test_cache_service.invalidate_cache("HR")

    # Now must MISS because KB version increased
    cached2, _, hit2 = test_cache_service.get_cached_answer(q, allowed_depts)
    assert cached2 is None
    assert hit2 == "MISS"
