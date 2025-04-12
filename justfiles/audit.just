# check number of lines in cursor rule files
[group('audit')]
check-rule-lines:
    ./scripts/check_rule_lines.py .cursor/rules/

# audit cursor rules in staging environment
[group('audit')]
audit-cursor-rules-stage:
    uv run scripts/audit_cursor_rules_headers.py

# audit cursor rules in staging environment with descriptions
[group('audit')]
audit-cursor-rules-stage-desc:
    uv run scripts/audit_cursor_rules_headers.py --desc

# audit cursor rules in production environment
[group('audit')]
audit-cursor-rules-prod:
    uv run scripts/audit_cursor_rules_headers.py --prod

# audit cursor rules in production environment with descriptions
[group('audit')]
audit-cursor-rules-prod-desc:
    uv run scripts/audit_cursor_rules_headers.py --prod --desc

# generate mermaid diagram of cursor rules relationships
[group('audit')]
generate-rules-diagram:
    ./scripts/generate_rule_mermaid.py

# generate mermaid diagram of cursor rules relationships (production)
[group('audit')]
generate-rules-diagram-prod:
    ./scripts/generate_rule_mermaid.py --prod
