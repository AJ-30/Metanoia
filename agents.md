
### Essential rules

1. Provide detailed inline comments wherever there is any complex logic
2. Add type hints and return type annotations to every function and every important variable
3. Add class comments, module level comments
4. Add function docstring clearly explaining and stating the types, names and details of every parameter and return type.
5. Functions should not be very long. They should do one thing only.
6. Avoid very short micro-level functions.
7. Code should be easy to change.
8. Dont hardcode any constants in code. Put them in separate TOML files
9. Environment should configurable to easily run on a different device with minimum changes. Use setup scripts for that
10. Add readme files whever applicable
11. Change documentation and comments whenever changing functionality
12. Ask questions whenever things are unclear. Dont assume things.
13. Always Keep it Simple and Silly
14. Prioritize ease of use
15. Prioritize ease of change
16. Prioritize reproducibility

### ML specific rules

1. Prevent data leakage at any cost
2. Telemetry and visibility of telemetry is of utmost importance to the developer while training the model
3. Each aspect of telemetry should be clearly visible and readable by the developer so that developer can take decisions
4. Each run should be reproducible
5. Proper logging should be done for everything using our custom logger

## Mandatory Rules

### Mandatory File or Class Scope Rules

- **Respect module boundaries and layering**
  - Do not move files unless explicitly instructed.
  - Keep configuration separate from runtime logic.
  - Avoid circular dependencies.
- **Do not generate code that violates Liskov Substitution Principle (LSP)**
  - If a class derives from another class, analyze the assumptions of the base class before changing derived behavior.
  - Where function names are the same in base and derived classes, assume the derived function overrides the base function.
  - Preconditions of the derived method must be no stricter than the base method.
  - Postconditions of the derived method must be at least as strong as the base method.
  - If the preconditions or postconditions are unclear, ask the user to verify them.
- **Handle errors and exceptions without suppressing them**
  - Do not swallow exceptions silently.
  - Do not create empty catch blocks.
  - Catch specific exception types instead of the language's root exception type.
  - Do not catch catch-all forms such as `...`.
  - Catch exceptions as late as practical, usually in higher-level public entry points.
- **Read the relevant task context before generating or reviewing code**
  - Read the spec, task description, and relevant code comments before making changes.
  - If you are uncertain about something, say so and suggest investigation rather than guessing.

### Mandatory Function Scope Rules

- **Keep functions small and single-purpose**
- **Fail fast on invalid assumptions**
  - Return errors early within the function scope.
  - Assert assumptions about inputs and outputs where appropriate.
  - Throw exceptions as early as practical in the program logic.
- **Do not create functions that are easy to misuse**
  - Functions and classes must be easy to use and hard to misuse.
  - Implicit assumptions increase misuse risk and should be surfaced or removed.

## Preferred Guidelines

### Preferred File or Class Scope Guidelines

- **Minimize mutable global state**
  - Global variables, class statics, file-level statics, namespace-level statics, and singleton-style objects are global state.
  - Avoid adding new mutable global state.
  - Immutable global state is acceptable when justified.
- **Minimize dependencies on external files and classes**
  - Imports, includes, and using statements indicate dependencies.
  - Avoid adding new dependencies while modifying an existing file unless they are necessary.
  - If a new dependency is required, explain why.
  - When generating a new file, minimize the number of external dependencies.
  - Dependency injection is still a runtime dependency; prefer simpler compile-time structure where appropriate.
- **Prefer Tell, Don't Ask for class design**
  - Class methods should tell the class to do useful work on its own state.
  - Minimize getters that only expose internal state.
  - Minimize setters that only mutate fields directly.
  - Do not return direct pointers or references to internal collections.

### Preferred Function Scope Guidelines

- **Keep function complexity under control**
  - Prefer functions that are short enough to understand without scrolling.
  - Prefer cyclomatic complexity below 10.
  - Prefer block depth below 6.
  - The 25-line guideline is a heuristic, not a rigid limit.
- **Make assumptions and side effects explicit**
  - Prefer explicit code over clever or implicit behavior.
  - Do not make implicit assumptions about inputs or callee behavior.
  - Avoid hard-coded values for array bounds, flags, and similar control data.
  - Use annotations or decorators where available to document assumptions.
  - Where annotations are unavailable, prefer assertions or equivalent guard logic.
  - Watch for unintended side effects, especially overwrites, aliasing, and lifetime issues.
- **Prefer Law of Demeter friendly code**
  - Avoid digging through the internals of another object.
  - Avoid long chains of getter calls.
  - Builder-style or fluent APIs are acceptable when they are the intended interface.

### Additional Guidelines for Code Review

- `Review Only` **Check for dead code or redundant code**
  - Dead code is code that exists in source files but is not called from anywhere.
  - This may be a private function with no callers or a variable that is declared but never used.
  - Use compiler warnings and static analysis where available.
- `Review Only` **Detect code duplication**
  - Check for exact duplicate code in multiple functions.
  - Duplication can exist in the same file or across multiple files.
  - Use tools such as CPD where available.
  - Highlight duplication as a potential design issue.
- Diff Review
  - Identify the differences between two given revisions or latest revision of branch and master/main branch or two different branches.
  - Generate a 'diff view' of changes between these revisions.
  - Use 'unified diff' format for generating the 'diff views'
  - Use the information in 'version control skill' to generate the diff or 'diff view'
  - Analyze the diff view using the mandatory, preferred, and review-only guidelines defined in this file.
  - "New dependency" added is important check in 'diff review'
- Periodically do a full file review applying all the guidelines.

## Additional Guidelines for Writing Unit Tests

- Use the Arrange-Act-Assert (AAA) pattern.
- Test happy paths, boundary values, null/undefined inputs, and expected exceptions.
- Use specification documents, design documents and test plans to identify conditions to test.
- Mock all external dependencies and network calls, database calls
- Name tests clearly to describe the scenario and expected outcome.
- Do not include placeholder comments; write fully working code.
