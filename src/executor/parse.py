from src.data.format import parse_action_json, parsed_action_name

def parse_tool_call(text):
    return parse_action_json(text)

def validate_tool_call(parsed, *, allowed_actions=None):
    if not parsed:
        return False
    action = parsed_action_name(parsed)
    if not action:
        return False

    if allowed_actions and action not in allowed_actions and (action != "noop"):
        return False
    if "actions" in parsed:
        return isinstance(parsed.get("actions"), list)
    args = parsed.get("arguments")
    return isinstance(args, dict)
