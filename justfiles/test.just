
# test code
[group('test')]
test:
	uv run pytest -s --verbose --showlocals --tb=short {{TESTS}}

# test code with debug
[group('test')]
test-debug:
	uv run pytest -s --verbose --showlocals --tb=short --pdb --pdbcls bpdb:BPdb {{TESTS}}
