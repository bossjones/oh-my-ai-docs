# Cursor Rules Reference

This document provides a comprehensive reference for all Cursor rules in the project. Each rule is designed to govern specific aspects of development, ensuring consistency and quality across the codebase.

## Core Rules

### [Rule Generation](../.cursor/rules/core-rules/rule-generating-agent.mdc)
This rule is essential for maintaining consistency and quality in rule creation across the codebase. Apply this rule when:
- Creating new rules
- Modifying existing rules
- Implementing behavior patterns that need to be remembered
- Requesting future behavior changes

### [Diagram Generation](../.cursor/rules/core-rules/diagram-generation-agent.mdc)
Governs the generation and management of Mermaid diagrams from stories and tests. Apply this rule when:
- Analyzing story files in .ai/stories/
- Analyzing Python test files
- Generating or updating diagrams in .ai/diagrams/
- Reviewing relationships between business logic and code dependencies

### [Project Status Tracking](../.cursor/rules/core-rules/project-status-tracker-agent.mdc)
Governs project status tracking and reporting. Apply this rule when:
- Requesting status checks
- Evaluating progress against story files
- Planning new implementation tasks
- Verifying project structure

### [Development Environment](../.cursor/rules/core-rules/development-environment-agent.mdc)
Governs development environment standards. Apply this rule when:
- Setting up development environments
- Discussing IDE configurations
- Handling environment variables
- Discussing deployment environments

### [Epic Story Management](../.cursor/rules/core-rules/epic-story-management-agent.mdc)
Governs the management of epics and stories. Apply this rule when:
- Creating or updating stories
- Managing epic status and progress
- Transitioning between stories
- Validating story completeness

### [Phased Development](../.cursor/rules/core-rules/phased-development-agent.mdc)
Governs the phased development approach. Apply this rule when:
- Creating new features or modules
- Modifying existing code
- Reviewing implementation details
- Planning development tasks

### [Phased Implementation](../.cursor/rules/core-rules/phased-implementation-agent.mdc)
Governs feature implementation based on phase status. Apply this rule when:
- Implementing new features from stories
- Modifying existing features
- Adding functionality to existing components
- Reviewing implementation tasks

### [PRD Section Navigation](../.cursor/rules/core-rules/prd-section-navigation-agent.mdc)
Governs efficient navigation of PRD documents. Apply this rule when:
- Searching for specific PRD sections
- Reading or updating PRD content
- Analyzing PRD sections for completeness
- Referencing PRD content during development

### [Security Monitoring](../.cursor/rules/core-rules/security-monitoring-agent.mdc)
Governs security and monitoring standards. Apply this rule when:
- Implementing security measures
- Setting up monitoring
- Handling rate limits
- Implementing error handling

### [Secure Environment Testing](../.cursor/rules/testing-rules/secure-environment-testing-agent.mdc)
Governs secure environment variable handling in tests. Apply this rule when:
- Writing tests that involve environment variables
- Handling sensitive data in tests
- Implementing settings validation tests
- Working with secret management

## Testing Rules

### [Pytest Mock Auto](../.cursor/rules/py-rules/pytest-mock-auto.mdc)
Automatically enforces pytest-mock standards in test files. This rule:
- Prevents usage of unittest.mock
- Enforces pytest-mock's mocker fixture
- Ensures proper mock cleanup
- Maintains consistent mocking patterns

### [Pytest Mock Agent](../.cursor/rules/py-rules/pytest-mock-agent.mdc)
Governs the use of mocking in Python tests. Apply this rule when:
- Writing new test files
- Modifying existing tests with mocks
- Reviewing test code with mocks
- Converting tests to use pytest-mock

## Global Rules

### [Emoji Communication](../.cursor/rules/global-rules/emoji-communication-always.mdc)
Governs the use of emojis in communication. This rule is always applied and ensures:
- Purposeful emoji usage to enhance meaning
- Professional yet engaging communication
- Consistent emoji placement and frequency
- Contextually appropriate emoji choices

## Python Rules

### [Python TDD](../.cursor/rules/py-rules/python-tdd-auto.mdc)
Governs Test-Driven Development workflow. Apply this rule when:
- Implementing new features
- Fixing bugs
- Refactoring code
- Working on story implementations

### [Pytest Fixture Naming](../.cursor/rules/py-rules/pytest-fixture-naming-agent.mdc)
Governs naming conventions for pytest fixtures to avoid collisions with built-ins, modules, and other Python entities. Apply this rule when:
- Generating code in the tests/*/*/* directory
- Ensuring fixture names are descriptive and scoped
- Avoiding naming collisions with Python built-ins and modules

## Tool Rules

### [Git Commit and Push](../.cursor/rules/tool-rules/git-commit-push-agent.mdc)
Governs git commit and push operations. Apply this rule when:
- Committing changes
- Pushing changes
- Following git commit conventions
- Updating work in git

### [LangGraph Documentation](../.cursor/rules/tool-rules/langgraph-docs-agent.mdc)
Governs handling of LangGraph documentation queries. Apply this rule when:
- Asking about LangGraph functionality
- Seeking clarification on LangGraph features
- Implementing LangGraph components
- Reviewing LangGraph documentation

### [UV Package Manager](../.cursor/rules/tool-rules/uv-package-manager-agent.mdc)
Governs UV package manager usage. Apply this rule when:
- Installing or managing Python packages
- Setting up Python environments
- Running Python code or tests
- Modifying dependency files

## TypeScript Rules

### [TypeScript Best Practices](../.cursor/rules/ts-rules/typescript-best-practices-agent.mdc)
Governs TypeScript development practices. Apply this rule when:
- Planning TypeScript features/components
- Modifying TypeScript code
- Reviewing/fixing TypeScript bugs
- Making architectural decisions
- Creating/refactoring TypeScript files

## UI Rules

### [UI Best Practices](../.cursor/rules/ui-rules/ui-best-practices-agent.mdc)
Governs UI development practices. Apply this rule when:
- Implementing user interfaces
- Designing UI components
- Ensuring accessibility standards
- Managing UI state

## Workflow Rules

### [Architecture Workflow](../.cursor/rules/workflows/arch.mdc)
Governs architectural decisions and workflows. Apply this rule when:
- Making architectural decisions
- Designing system components
- Planning system integrations
- Evaluating technical choices

### [Development Workflow](../.cursor/rules/workflows/dev.mdc)
Governs development workflows and practices. Apply this rule when:
- Following development processes
- Setting up development tasks
- Managing development cycles
- Coordinating development efforts

### [Project Management Workflow](../.cursor/rules/workflows/pm.mdc)
Governs project management practices. Apply this rule when:
- Managing project timelines
- Coordinating team efforts
- Tracking project progress
- Planning project milestones

## Rule Types

The project uses several types of rules:
- **Agent Rules** (`*-agent.mdc`): Rules that are applied by the AI when specific conditions are met
- **Auto Rules** (`*-auto.mdc`): Rules that automatically apply to files matching specific patterns
- **Manual Rules** (`*-manual.mdc`): Rules that must be explicitly invoked
- **Always Rules** (`*-always.mdc`): Rules that are always applied in every conversation
