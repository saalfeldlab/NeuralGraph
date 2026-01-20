#!/usr/bin/env python3
"""
Fix block numbering issue in signal_chaotic_1_Claude files.
- Remove iteration 128 (the orphan Block 9 with only 1 iteration)
- Renumber iterations 129-320 to 128-319
- Renumber blocks 10-22 to 9-21

Uses placeholder approach to avoid cascading replacements.
"""

import re

def fix_reasoning_log():
    """Fix the reasoning log file."""
    with open('signal_chaotic_1_Claude_reasoning.log', 'r') as f:
        content = f.read()

    # Step 1: Find and remove iteration 128 section BEFORE any renumbering
    # The section starts with "============================================================\n=== Iteration 128 ==="
    # and ends just before "============================================================\n=== Iteration 129 ==="

    # Find the start position
    start_marker = "============================================================\n=== Iteration 128 ===\n============================================================\n"
    end_marker = "============================================================\n=== Iteration 129 ==="

    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)

    if start_pos != -1 and end_pos != -1:
        # Remove everything from start_marker to just before end_marker
        content = content[:start_pos] + content[end_pos:]
        print(f"Removed iteration 128 section (chars {start_pos} to {end_pos})")
    else:
        print(f"Warning: Could not find iteration 128 section. start={start_pos}, end={end_pos}")

    # Step 2: Use placeholders to avoid cascading replacements
    # First pass: replace all numbers with placeholders
    for old_iter in range(129, 321):
        placeholder = f'__ITER_{old_iter}__'
        # Replace specific patterns with placeholders
        content = content.replace(f'=== Iteration {old_iter} ===', f'=== Iteration {placeholder} ===')
        content = re.sub(rf'(?<!\d)Iteration {old_iter}(?!\d)', f'Iteration {placeholder}', content)
        content = re.sub(rf'(?<![0-9])iter {old_iter}(?![0-9])', f'iter {placeholder}', content)
        content = re.sub(rf'(?<![0-9])node {old_iter}(?![0-9])', f'node {placeholder}', content)
        content = re.sub(rf'(?<![0-9])Node {old_iter}(?![0-9])', f'Node {placeholder}', content)
        content = content.replace(f'id={old_iter},', f'id={placeholder},')
        content = content.replace(f'id={old_iter})', f'id={placeholder})')
        content = content.replace(f'parent={old_iter},', f'parent={placeholder},')
        content = content.replace(f'parent={old_iter})', f'parent={placeholder})')
        content = content.replace(f'parent={old_iter} ', f'parent={placeholder} ')

    # Second pass: replace placeholders with new numbers
    for old_iter in range(129, 321):
        new_iter = old_iter - 1
        placeholder = f'__ITER_{old_iter}__'
        content = content.replace(placeholder, str(new_iter))

    # Step 3: Renumber blocks 10-22 to 9-21 using placeholders
    for old_block in range(10, 23):
        placeholder = f'__BLOCK_{old_block}__'
        content = re.sub(rf'(?<![0-9])Block {old_block}(?![0-9])', f'Block {placeholder}', content)

    for old_block in range(10, 23):
        new_block = old_block - 1
        placeholder = f'__BLOCK_{old_block}__'
        content = content.replace(placeholder, str(new_block))

    with open('signal_chaotic_1_Claude_reasoning.log', 'w') as f:
        f.write(content)

    print("Fixed reasoning log")


def fix_analysis_md():
    """Fix the analysis.md file."""
    with open('signal_chaotic_1_Claude_analysis.md', 'r') as f:
        content = f.read()

    # Use placeholders for iterations/nodes 129-320
    for old_iter in range(129, 321):
        placeholder = f'__ITER_{old_iter}__'
        content = content.replace(f'id={old_iter},', f'id={placeholder},')
        content = content.replace(f'id={old_iter})', f'id={placeholder})')
        content = content.replace(f'parent={old_iter},', f'parent={placeholder},')
        content = content.replace(f'parent={old_iter})', f'parent={placeholder})')
        content = content.replace(f'parent={old_iter} ', f'parent={placeholder} ')
        content = re.sub(rf'(?<![0-9])Iter {old_iter}:', f'Iter {placeholder}:', content)
        content = re.sub(rf'(?<![0-9])node {old_iter}(?![0-9])', f'node {placeholder}', content)
        content = re.sub(rf'(?<![0-9])Node {old_iter}(?![0-9])', f'Node {placeholder}', content)
        content = re.sub(rf'(?<![0-9])iter {old_iter}(?![0-9])', f'iter {placeholder}', content)
        content = re.sub(rf'(?<![0-9])iteration {old_iter}(?![0-9])', f'iteration {placeholder}', content)

    # Replace placeholders with new numbers
    for old_iter in range(129, 321):
        new_iter = old_iter - 1
        placeholder = f'__ITER_{old_iter}__'
        content = content.replace(placeholder, str(new_iter))

    # Renumber blocks 10-22 to 9-21 using placeholders
    for old_block in range(10, 23):
        placeholder = f'__BLOCK_{old_block}__'
        content = re.sub(rf'(?<![0-9])Block {old_block}(?![0-9])', f'Block {placeholder}', content)

    for old_block in range(10, 23):
        new_block = old_block - 1
        placeholder = f'__BLOCK_{old_block}__'
        content = content.replace(placeholder, str(new_block))

    with open('signal_chaotic_1_Claude_analysis.md', 'w') as f:
        f.write(content)

    print("Fixed analysis.md")


def fix_memory_md():
    """Fix the memory.md file."""
    with open('signal_chaotic_1_Claude_memory.md', 'r') as f:
        content = f.read()

    # Use placeholders for iterations 129-320
    for old_iter in range(129, 321):
        placeholder = f'__ITER_{old_iter}__'
        content = re.sub(rf'(?<![0-9])Iteration {old_iter}(?![0-9])', f'Iteration {placeholder}', content)
        content = re.sub(rf'(?<![0-9])iter {old_iter}(?![0-9])', f'iter {placeholder}', content)
        content = re.sub(rf'(?<![0-9])node {old_iter}(?![0-9])', f'node {placeholder}', content)
        content = re.sub(rf'(?<![0-9])Node {old_iter}(?![0-9])', f'Node {placeholder}', content)

    # Replace placeholders with new numbers
    for old_iter in range(129, 321):
        new_iter = old_iter - 1
        placeholder = f'__ITER_{old_iter}__'
        content = content.replace(placeholder, str(new_iter))

    # Renumber blocks 10-22 to 9-21 using placeholders
    for old_block in range(10, 23):
        placeholder = f'__BLOCK_{old_block}__'
        content = re.sub(rf'(?<![0-9])Block {old_block}(?![0-9])', f'Block {placeholder}', content)

    for old_block in range(10, 23):
        new_block = old_block - 1
        placeholder = f'__BLOCK_{old_block}__'
        content = content.replace(placeholder, str(new_block))

    with open('signal_chaotic_1_Claude_memory.md', 'w') as f:
        f.write(content)

    print("Fixed memory.md")


if __name__ == '__main__':
    fix_reasoning_log()
    fix_analysis_md()
    fix_memory_md()
    print("\nDone! Verify the changes and remove backups if satisfied.")
