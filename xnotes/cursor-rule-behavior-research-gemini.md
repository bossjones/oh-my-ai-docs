Original Question to Gemini 2.5 Pro preview: `USER: I would like some help updating rule-generating-agent.mdc based on all of the information I uploaded via files just now and this new comment that offers now insight into how cursor rules work in .cursor/rules/*.mdc. Think about things step by step and if you are not sure, then say that. Here is the new information:`


AI:

Okay, let's break this down and update the `rule-generating-agent.mdc.md` file based on the new insights from `cursor-rules-re.md` and the user's summary.

**Analysis of New Information:**

1.  **Rule Interpretation:** The core finding is how `description`, `globs`, and `alwaysApply` interact based on the actual code (`workbench.desktop.main.js` analysis).
    *   `alwaysApply: true`: Overrides everything. Rule is always active. `globs` are ignored for activation.
    *   `globs` (and `alwaysApply: false`): Rule auto-activates *only* when a relevant file (matching the glob) is in context. `description` doesn't affect this auto-activation but might be used by the UI or agent lookup.
    *   `description` (and `alwaysApply: false`, `globs` blank): Rule is *only* findable/selectable by the agent (e.g., via `@` mention or agent's internal logic). It does *not* auto-activate based on file context. `globs` cannot filter this type of agent-only rule activation.
    *   `description` *and* `globs` (and `alwaysApply: false`): This is a valid and useful combination! The rule is *both* agent-selectable (due to `description`) *and* auto-activates for matching files (due to `globs`). This corrects a previous misunderstanding.
    *   All blank / `alwaysApply: false`: Manual rule, only activated via explicit `@` reference.
2.  **Glob Format:** Confirmed: Comma-separated, no quotes. `src/**/*.ts, test/**/*.ts` is correct. `"src/**/*.ts, test/**/*.ts"` is incorrect.
3.  **User's Rule Type Logic:** The user's proposed priority (`alwaysApply` -> `description` -> `globs` -> `manual`) is a good simplification for *naming* and *categorizing* from a user perspective, even if the underlying mechanism allows combinations (like `description` + `globs`). We can adopt this for clarity in the generator's instructions.
4.  **Naming Convention:** The `rule-name-{agent|auto|always|manual}.mdc` convention is clear and useful.
5.  **Description Importance:** The user rightly emphasizes that `description` is key for agent understanding, especially for rules intended for agent selection.

**Updating `rule-generating-agent.mdc.md`:**

Based on this, we need to:

1.  Revise the "Critical Rules" section, especially the "FrontMatter Rules Types" part, to accurately reflect the four types based on the user's priority logic and the code's behavior.
2.  Clarify the behavior of combined `description` and `globs`.
3.  Reinforce the correct `globs` format.
4.  Incorporate the naming convention.
5.  Update the template comments and examples if necessary.

---

**Updated `rule-generating-agent.mdc.md`:**

```mdc
---
description: This rule is essential for maintaining consistency and quality in rule creation across the codebase. It must be followed whenever: (1) A user requests a new rule to be created, (2) An existing rule needs modification, (3) The user asks to remember certain behaviors or patterns, or (4) Future behavior changes are requested. This rule ensures proper organization, clear documentation, and effective rule application by defining standard formats, naming conventions, and content requirements based on Cursor v0.48.4+ behavior. It's particularly crucial for maintaining the rule hierarchy, ensuring rules are discoverable and applied correctly, and preserving the effectiveness of the rule-based system. The rule system is fundamental to project consistency, code quality, and automated assistance effectiveness.
globs:
alwaysApply: true
---
# Cursor Rules Format and Generation Guide (v0.48.4+)

## Template Structure for Rules Files (.mdc)

```mdc
---
description: |
  `Comprehensive description for agent understanding. Clearly state the rule's purpose and when it should apply. Crucial for 'Agent Selected' rules, helpful for others. Keep it focused.`
globs: `Comma-separated glob patterns (e.g., src/**/*.ts, *.md) OR blank. NO quotes around the list. Crucial for 'Auto Select' rules.`
alwaysApply: {true or false} # `true` makes it an 'Always' rule, ignoring globs for activation. `false` for all other types.
---

# Rule Title

## Critical Rules / Guidance

- Concise, bulleted list of actionable rules the agent MUST follow.
- Focus on clear directives.

## Examples

<example>
  # Valid Example: Shows correct application of the rule.
  # Indent content within example tags with 2 spaces.
  def correct_function():
    """Does the right thing."""
    pass # Complies with the rule
</example>

<example type="invalid">
  # Invalid Example: Shows incorrect application.
  # Indent content within example tags with 2 spaces.
  def incorrect_function():
    # Missing docstring, violates the rule
    pass
</example>
```

### Organizational Folders (Create if non-existent)
All rules files MUST be placed under an organizational subfolder within `.cursor/rules/`:
- `.cursor/rules/core-rules` - Rules about Cursor agent behavior or rule generation.
- `.cursor/rules/my-rules` - (gitignore recommended) Rules specific to an individual user.
- `.cursor/rules/global-rules` - Contains 'Always' type rules.
- `.cursor/rules/testing-rules` - Rules about testing frameworks, coverage, etc.
- `.cursor/rules/tool-rules` - Rules for specific tools (git, linters, MCP).
- `.cursor/rules/ts-rules` - TypeScript specific rules.
- `.cursor/rules/py-rules` - Python specific rules.
- `.cursor/rules/ui-rules` - Rules about HTML, CSS, React, etc.
* Create new folders under `.cursor/rules/` as needed, following similar logical groupings (e.g., `.cursor/rules/java-rules`).

## Glob Pattern Examples
Common glob patterns (remember: comma-separated, no quotes):
- Language rules: `*.ts, *.tsx`
- Testing standards: `*.test.ts, *.spec.js`
- React components: `src/components/**/*.tsx`
- Documentation: `docs/**/*.md, *.md`
- Configuration files: `*.config.js, *.json`
- Multiple patterns: `dist/**/*.*, docs/**/*.md, *test*.*`

## Critical Rules for Rule Generation

1.  **Location and Naming:**
    *   Rule files MUST be located in `.cursor/rules/{organizational-folder}/`.
    *   Rule files MUST be named following the pattern: `rule-name-{type}.mdc`, where `{type}` is one of `agent`, `always`, `auto`, or `manual`. Use dashes `-` to separate words in `rule-name`.
    *   Always check existing rules in all `.cursor/rules/` subfolders before creating a new one; update existing rules if appropriate.

2.  **FrontMatter and Rule Type Determination:**
    *   The frontmatter section (`--- ... ---`) MUST start the file and contain all three fields: `description`, `globs`, and `alwaysApply`.
    *   Determine the rule type based on the following priority:
        *   **Always Rule:** If `alwaysApply: true`.
            *   `description`: Can be blank or descriptive (useful for UI).
            *   `globs`: Can be blank or populated, but **will be ignored** for activation purposes.
            *   Filename MUST end with `-always.mdc`.
            *   *Behavior:* Applied to every chat and command request, regardless of file context. Place in `global-rules` folder.
        *   **Agent Selected Rule:** If `alwaysApply: false` AND `description` is non-empty AND `globs` is blank.
            *   `description`: MUST be comprehensive, providing context for the agent to decide when to apply it.
            *   `globs`: MUST be blank.
            *   `alwaysApply`: MUST be `false`.
            *   Filename MUST end with `-agent.mdc`.
            *   *Behavior:* Not applied automatically. The agent uses the description to decide if it's relevant to the current task/conversation.
        *   **Auto Select Rule:** If `alwaysApply: false` AND `globs` is non-empty.
            *   `description`: Can be blank, or descriptive (recommended for clarity and potential agent use).
            *   `globs`: MUST contain valid, comma-separated glob patterns (NO quotes).
            *   `alwaysApply`: MUST be `false`.
            *   Filename MUST end with `-auto.mdc`.
            *   *Behavior:* Automatically applied when a file matching the `globs` is in context (e.g., open file, file mentioned in chat). If `description` is also provided, the agent *can also* select it like an 'Agent Selected' rule.
        *   **Manual Rule:** If `alwaysApply: false` AND `description` is blank AND `globs` is blank.
            *   `description`: MUST be blank.
            *   `globs`: MUST be blank.
            *   `alwaysApply`: MUST be `false`.
            *   Filename MUST end with `-manual.mdc`.
            *   *Behavior:* Never applied automatically. Only used when explicitly referenced by the user (e.g., `@my-manual-rule`).

3.  **Glob Format:**
    *   `globs` value MUST be a comma-separated list of patterns.
    *   NEVER enclose the entire list or individual patterns in quotes (e.g., use `src/*.ts, test/*.ts`, NOT `"src/*.ts, test/*.ts"`).

4.  **Rule Content:**
    *   Focus on actionable, clear directives in the main body.
    *   Use concise Markdown. Emojis and Mermaid diagrams are acceptable if they enhance clarity without redundancy.
    *   Be mindful of length; prioritize essential information.
    *   ALWAYS include one valid `<example>` and one invalid `<example type="invalid">`. Indent content within example tags by 2 spaces.
    *   If the rule request stems from correcting a mistake, use that context in the examples.

5.  **Response After Creation/Update:**
    *   Respond with:
        ```
        AutoRuleGen Success: .cursor/rules/{folder}/rule-name-{type}.mdc
        Rule Type: {Always | Agent Selected | Auto Select | Manual}
        Description: {Exact content of the description field, or 'N/A' if blank}
        Globs: {Exact content of the globs field, or 'N/A' if blank}
        AlwaysApply: {true | false}
        ```

```

**Summary of Key Changes:**

*   Updated the main description to mention v0.48.4+ and the source of the info.
*   Refined the template comments for clarity on `description`, `globs`, and `alwaysApply`.
*   Rewrote the "FrontMatter and Rule Type Determination" section based on the 4 types derived from the user's priority and code behavior.
*   Explicitly defined the behavior of each rule type.
*   Clarified the valid and useful combination of `description` and `globs` for the "Auto Select" type.
*   Reinforced the correct `globs` format (comma-separated, no quotes).
*   Incorporated the file naming convention (`rule-name-{type}.mdc`).
*   Updated the example structure slightly for clarity.
*   Adjusted the success response format to include all key fields.

This updated rule should now guide the agent to create `.mdc` rules that align correctly with how Cursor v0.48.4+ interprets them.
