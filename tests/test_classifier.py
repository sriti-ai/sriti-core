import pytest
from sriti.core.cascade import classifier


def test_classifier_warm_up_and_classify():
    classifier.warm_up()
    messages = [{"role": "user", "content": "Can you summarize our Q3 earnings report?"}]
    task_type, confidence, emb = classifier.classify(messages)
    assert task_type in ("summarization", "instruction_following", "rag_retrieval", "document_review")
    assert confidence > 0.0
    assert len(emb) == 384


def test_classifier_code_detection():
    classifier.warm_up()
    messages = [{"role": "user", "content": "def bubble_sort(arr):\n    for i in range(len(arr)):\n        pass"}]
    task_type, confidence, emb = classifier.classify(messages)
    assert task_type == "code"
    assert confidence > 0.0
