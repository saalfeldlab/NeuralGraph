#!/usr/bin/env python3
"""
Complete fix for block numbering issue in signal_chaotic_1_Claude files.
- Remove iteration 128 (the orphan Block 9 with only 1 iteration)
- Renumber iterations 129-320 to 128-319 (including all prose references)
- Renumber blocks 10-22 to 9-21, and Block 23 to Block 22

Uses placeholder approach to avoid cascading replacements.
"""

import re

def fix_file(filename, remove_iter_128=False):
    """Fix a single file."""
    with open(filename, 'r') as f:
        content = f.read()

    if remove_iter_128:
        # Step 1: Find and remove iteration 128 section BEFORE any renumbering
        start_marker = "============================================================\n=== Iteration 128 ===\n============================================================\n"
        end_marker = "============================================================\n=== Iteration 129 ==="

        start_pos = content.find(start_marker)
        end_pos = content.find(end_marker)

        if start_pos != -1 and end_pos != -1:
            content = content[:start_pos] + content[end_pos:]
            print(f"  Removed iteration 128 section")
        else:
            print(f"  Warning: Could not find iteration 128 section")

    # Step 2: Replace ALL iteration numbers 129-320 with placeholders
    # This includes headers, prose text, node references, etc.
    for old_iter in range(129, 321):
        placeholder = f'__ITER_{old_iter}__'

        # Headers
        content = content.replace(f'=== Iteration {old_iter} ===', f'=== Iteration {placeholder} ===')

        # Prose patterns - be comprehensive
        content = re.sub(rf'(?<![0-9])Iteration {old_iter}(?![0-9])', f'Iteration {placeholder}', content)
        content = re.sub(rf'(?<![0-9])iteration {old_iter}(?![0-9])', f'iteration {placeholder}', content)
        content = re.sub(rf'(?<![0-9])iter {old_iter}(?![0-9])', f'iter {placeholder}', content)
        content = re.sub(rf'(?<![0-9])Iter {old_iter}(?![0-9])', f'Iter {placeholder}', content)

        # Node references
        content = re.sub(rf'(?<![0-9])node {old_iter}(?![0-9])', f'node {placeholder}', content)
        content = re.sub(rf'(?<![0-9])Node {old_iter}(?![0-9])', f'Node {placeholder}', content)

        # ID references
        content = content.replace(f'id={old_iter},', f'id={placeholder},')
        content = content.replace(f'id={old_iter})', f'id={placeholder})')
        content = content.replace(f'id={old_iter} ', f'id={placeholder} ')
        content = content.replace(f'parent={old_iter},', f'parent={placeholder},')
        content = content.replace(f'parent={old_iter})', f'parent={placeholder})')
        content = content.replace(f'parent={old_iter} ', f'parent={placeholder} ')

        # "setup for X" patterns
        content = re.sub(rf'setup for {old_iter}(?![0-9])', f'setup for {placeholder}', content)
        content = re.sub(rf'for {old_iter}(?![0-9])', f'for {placeholder}', content)

    # Step 3: Replace placeholders with new numbers
    for old_iter in range(129, 321):
        new_iter = old_iter - 1
        placeholder = f'__ITER_{old_iter}__'
        content = content.replace(placeholder, str(new_iter))

    # Step 4: Renumber blocks 10-23 to 9-22 using placeholders
    for old_block in range(10, 24):
        placeholder = f'__BLOCK_{old_block}__'
        content = re.sub(rf'(?<![0-9])Block {old_block}(?![0-9])', f'Block {placeholder}', content)

    for old_block in range(10, 24):
        new_block = old_block - 1
        placeholder = f'__BLOCK_{old_block}__'
        content = content.replace(placeholder, str(new_block))

    with open(filename, 'w') as f:
        f.write(content)


def main():
    print("Fixing reasoning log...")
    fix_file('/workspace/NeuralGraph/signal_chaotic_1_Claude_reasoning.log', remove_iter_128=True)

    print("Fixing analysis.md...")
    fix_file('/workspace/NeuralGraph/signal_chaotic_1_Claude_analysis.md', remove_iter_128=False)

    print("Fixing memory.md...")
    fix_file('/workspace/NeuralGraph/signal_chaotic_1_Claude_memory.md', remove_iter_128=False)

    print("\nDone!")


if __name__ == '__main__':
    main()
