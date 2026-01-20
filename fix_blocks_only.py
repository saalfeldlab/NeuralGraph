#!/usr/bin/env python3
"""
Simple fix: decrease block numbers > 9 by one.
Block 10 -> Block 9, Block 11 -> Block 10, etc.
Does NOT change iteration numbers.
"""

import re

def fix_blocks(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Use placeholders to avoid cascading replacements
    # Replace Block 10-23 with placeholders
    for old_block in range(10, 24):
        placeholder = f'__BLOCK_{old_block}__'
        content = re.sub(rf'\bBlock {old_block}\b', f'Block {placeholder}', content)

    # Replace placeholders with new numbers (old - 1)
    for old_block in range(10, 24):
        new_block = old_block - 1
        placeholder = f'__BLOCK_{old_block}__'
        content = content.replace(f'Block {placeholder}', f'Block {new_block}')

    with open(filename, 'w') as f:
        f.write(content)

    print(f"Fixed {filename}")

if __name__ == '__main__':
    fix_blocks('/workspace/NeuralGraph/signal_chaotic_1_Claude_reasoning.log')
    fix_blocks('/workspace/NeuralGraph/signal_chaotic_1_Claude_analysis.md')
    fix_blocks('/workspace/NeuralGraph/signal_chaotic_1_Claude_memory.md')
    print("\nDone! Block numbers > 9 decreased by 1.")
