from __future__ import annotations
import pytest

def test_controller_preserves_stage_order():
    from importlib import import_module
    mod=import_module("06_COGNITION.cognitive_controller")
    stages=mod.CognitiveController((lambda x:x+1,lambda x:x*2)).run(2)
    assert [s.output for s in stages]==[3,6]

def test_attention_selects_lexical_match():
    from importlib import import_module
    fn=import_module("06_COGNITION.attention_mechanism").select_focus
    assert fn("cooling system",["unrelated","cooling system inspection"])[0][2]=="cooling system inspection"

def test_abstraction_counts_terms():
    from importlib import import_module
    fn=import_module("06_COGNITION.abstraction_engine").extract_terms
    assert fn(["alpha beta alpha"])[0]==("alpha",2)

def test_concept_requires_evidence():
    from importlib import import_module
    mod=import_module("06_COGNITION.concept_formation")
    with pytest.raises(ValueError): mod.form_concept("pump",[])
    assert mod.form_concept("pump",["e1"]).evidence_ids==("e1",)

def test_relationship_preserves_evidence():
    from importlib import import_module
    rel=import_module("06_COGNITION.relationship_engine").build_relation("a","connected_to","b","e1")
    assert rel.evidence_id=="e1"

def test_causal_reachability_only_traverses_given_edges():
    from importlib import import_module
    fn=import_module("06_COGNITION.causal_reasoning").causal_reachability
    assert fn([("a","b"),("b","c")],"a")==("b","c")

def test_logic_primitives():
    from importlib import import_module
    mod=import_module("06_COGNITION.logical_reasoning")
    assert mod.modus_ponens(True,True) is True
    assert mod.modus_ponens(False,True) is None

def test_safe_math():
    from importlib import import_module
    fn=import_module("06_COGNITION.mathematical_reasoning").calculate
    assert fn(6,"*",7)==42
    with pytest.raises(ZeroDivisionError): fn(1,"/",0)
    with pytest.raises(ValueError): fn(2,"import",3)

def test_hypothesis_requires_evidence_context():
    from importlib import import_module
    mod=import_module("06_COGNITION.hypothesis_generation")
    with pytest.raises(ValueError): mod.propose("pump failure")
    assert mod.propose("pump cavitation",missing_evidence=("pressure log",)).statement=="pump cavitation"

def test_uncertainty_entropy():
    from importlib import import_module
    mod=import_module("06_COGNITION.uncertainty_estimation")
    assert mod.normalized_entropy([1,1])==pytest.approx(1.0)
    assert mod.normalized_entropy([1,0])==pytest.approx(0.0)

def test_metacognitive_audit():
    from importlib import import_module
    from types import SimpleNamespace
    mod=import_module("06_COGNITION.metacognition")
    result=SimpleNamespace(evidence=[1],decision=SimpleNamespace(status=SimpleNamespace(value="abstained"),rationale="insufficient evidence"),warnings=())
    audit=mod.audit(result)
    assert audit.evidence_count==1 and audit.decision_abstained and audit.has_rationale
