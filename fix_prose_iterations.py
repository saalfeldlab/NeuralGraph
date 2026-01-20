#!/usr/bin/env python3
"""
Fix prose iteration references in reasoning log.
These are text like "Summary of iteration 129:" that should now be "Summary of iteration 128:"
"""

import re

def fix_prose_iterations(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Patterns to fix - prose text that references iteration numbers
    # These are inside sentences, not the header markers

    patterns = [
        (r'Summary of iteration (\d+)', 'Summary of iteration'),
        (r'summary of iteration (\d+)', 'summary of iteration'),
        (r'for iteration (\d+)', 'for iteration'),
        (r'setup for (\d+)', 'setup for'),
        (r'iteration (\d+) complete', 'iteration'),  # will append ' complete'
    ]

    # Use placeholder approach
    changes = []

    # Find all "Summary of iteration XXX" patterns where XXX >= 129
    for match in re.finditer(r'Summary of iteration (\d+)', content):
        old_num = int(match.group(1))
        if old_num >= 129:
            changes.append((match.start(), match.end(), old_num, old_num - 1, match.group(0)))

    # Find "summary of iteration XXX" (lowercase)
    for match in re.finditer(r'summary of iteration (\d+)', content):
        old_num = int(match.group(1))
        if old_num >= 129:
            changes.append((match.start(), match.end(), old_num, old_num - 1, match.group(0)))

    # Find "for iteration XXX"
    for match in re.finditer(r'for iteration (\d+)', content):
        old_num = int(match.group(1))
        if old_num >= 129:
            changes.append((match.start(), match.end(), old_num, old_num - 1, match.group(0)))

    # Find "setup for XXX" (just the number after "setup for")
    for match in re.finditer(r'setup for (\d+)', content):
        old_num = int(match.group(1))
        if old_num >= 129:
            changes.append((match.start(), match.end(), old_num, old_num - 1, match.group(0)))

    # Sort by position in reverse order so we can replace without messing up positions
    changes.sort(key=lambda x: x[0], reverse=True)

    # Apply changes
    for start, end, old_num, new_num, original in changes:
        new_text = original.replace(str(old_num), str(new_num))
        content = content[:start] + new_text + content[end:]
        print(f"Fixed: '{original}' -> '{new_text}'")

    with open(filename, 'w') as f:
        f.write(content)

    print(f"\nTotal fixes: {len(changes)}")

if __name__ == '__main__':
    fix_prose_iterations('/workspace/NeuralGraph/signal_chaotic_1_Claude_reasoning.log')
