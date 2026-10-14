/// Chat completions route on their message text when no session_id is given, so cache_aware can match a
/// conversation turn against the prefix of the previous one.
use vllm_router_rs::protocols::spec::{ChatCompletionRequest, GenerationRequest};

fn chat(json: &str) -> ChatCompletionRequest {
    serde_json::from_str(json).unwrap()
}

#[test]
fn test_chat_routing_text_uses_messages() {
    let req = chat(
        r#"{"model": "m", "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": [{"type": "text", "text": "List the files."}]},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
              "function": {"name": "ls", "arguments": "{\"path\": \".\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "a.py b.py"},
            {"role": "user", "content": "Open a.py"}
        ]}"#,
    );
    let text = req.extract_text_for_routing();
    assert!(text.starts_with("You are a coding agent.\nList the files.\n"));
    assert!(text.contains("\"name\":\"ls\""));
    assert!(text.ends_with("a.py b.py\nOpen a.py"));
}

#[test]
fn test_chat_routing_text_next_turn_extends_previous() {
    let turn1 = chat(r#"{"model": "m", "messages": [{"role": "user", "content": "hi"}]}"#);
    let turn2 = chat(
        r#"{"model": "m", "messages": [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]}"#,
    );
    assert!(turn2
        .extract_text_for_routing()
        .starts_with(&turn1.extract_text_for_routing()));
}

#[test]
fn test_chat_routing_text_prefers_session_id() {
    let req = chat(
        r#"{"model": "m", "messages": [{"role": "user", "content": "hi"}],
            "session_params": {"session_id": "s-1"}}"#,
    );
    assert_eq!(req.extract_text_for_routing(), "s-1");
}
