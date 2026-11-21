from langchain_core.messages import AIMessage

from app.graph.build_graph import build_graph
from app.graph.state import Plan, Reflection
from eval.run_eval import run_task, score
from tests.conftest import tc


def test_score_groups_and_files():
    t = {"expect_any_of": [["hmac"], ["signer.py", "sign.py"]], "expect_files": ["a/signer.py"]}
    assert score(t, {"final_answer": "HMAC in Signer.py", "files_read": ["a/signer.py"]})
    assert not score(t, {"final_answer": "HMAC in Signer.py", "files_read": []})
    assert not score(t, {"final_answer": "only hmac", "files_read": ["a/signer.py"]})


def test_run_task_auto_rejects_writes(ctx, make_llm, saver, fake_gh):
    llm, _ = make_llm(
        responses=[AIMessage("", tool_calls=[tc("create_issue", {"title": "x", "body": "y"}, "w1")]),
                   AIMessage("The issue was rejected by the reviewer.")],
        structured=[Plan(steps=["issue"]), Reflection(complete=True, grounded=True)])
    g = build_graph(ctx, llm, checkpointer=saver, enable_tests=False)
    r = run_task(g, {"id": "issue_rejected", "task": "open issue", "approve": False,
                     "expect_any_of": [["reject"]]})
    assert r["passed"] and r["tool_calls"] == 1 and fake_gh.issues == []
