Rule Type Determination Function
[x] Update determine_rule_type() to include file suffix checking for proper naming convention
[x] Add checks for -agent.mdc, -auto.mdc, -always.mdc, and -manual.mdc suffixes
[x] Validate that the file name matches the determined rule type
[x] Add rule type identification based on file suffix in addition to frontmatter
Rule Explanation and Examples
[x] Update print_rule_type_examples() to include more detailed information about each rule type
[x] Add examples showing the required file suffix pattern for each rule type
[x] Include information about different organizational folders
[x] Add examples of valid file names with proper suffixes
File Detection and Organization
[x] Update audit_cursor_rules() to check for proper organizational folder structure
[x] Validate that rules are placed in the appropriate organizational folders
[x] Check if folder structure follows the convention listed in documentation
[x] Add detection for rules in organizational folders like core-rules, ts-rules, etc.
Directory Structure
[x] Update default directory paths to include proper organizational subfolders
[x] Add detection for common organizational folders mentioned in the documentation
[x] Add validation for file placement within the organizational structure
Type-specific Validation
[x] Add specific checks for each rule type:
[x] Always rules: validate file name ends with -always.mdc
[x] Agent Selected rules: validate file name ends with -agent.mdc
[x] Auto Select rules: validate file name ends with -auto.mdc
[x] Manual rules: validate file name ends with -manual.mdc
Type Definitions
[x] Fix type annotations to address linter errors
[x] Ensure proper typing for all functions
[x] Add more specific type hints for rule types
Documentation Updates
[x] Update docstrings to reflect new understanding of rule types
[x] Include information about file naming conventions
[x] Add more details about organizational folders
Output Improvements
[x] Add information about file naming conventions in the output
[x] Include visualization of organizational folder structure
[x] Improve error messages to suggest correct folder placement
All tasks completed! The script has been updated to include file suffix checking, organizational folder validation, and improved type annotations.
