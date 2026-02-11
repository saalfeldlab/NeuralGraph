#!/bin/bash
# Render the Quarto site and compress all PNGs to fit under GitHub's 500KB limit.
set -e

SITE_DIR="/workspace/NeuralGraph"
MAX_KB=499
JOBS=16  # parallel compression jobs (machine has 64 cores)

cd "$SITE_DIR"

echo "=== Cleaning build artifacts ==="
rm -rf .quarto _freeze docs

echo "=== Running quarto render ==="
quarto render

# ---------------------------------------------------------------------------
# Compress a single PNG to fit under MAX_KB.
# Called by xargs in parallel.
# ---------------------------------------------------------------------------
compress_png() {
    local f="$1"
    local MAX_KB="$2"
    local orig_kb
    orig_kb=$(du -k "$f" | cut -f1)

    # 1. Strip metadata + iterative quality reduction
    local quality=90
    while [ "$quality" -ge 10 ]; do
        mogrify -strip -quality "$quality" "$f"
        local new_kb
        new_kb=$(du -k "$f" | cut -f1)
        if [ "$new_kb" -le "$MAX_KB" ]; then
            echo "  $f: ${orig_kb}KB -> ${new_kb}KB (q=$quality)"
            return
        fi
        quality=$((quality - 10))
    done

    # 2. Resize 50%
    mogrify -strip -resize 50% "$f"
    new_kb=$(du -k "$f" | cut -f1)
    if [ "$new_kb" -le "$MAX_KB" ]; then
        echo "  $f: ${orig_kb}KB -> ${new_kb}KB (resized 50%)"
        return
    fi

    # 3. Resize 25%
    mogrify -strip -resize 25% "$f"
    new_kb=$(du -k "$f" | cut -f1)
    if [ "$new_kb" -le "$MAX_KB" ]; then
        echo "  $f: ${orig_kb}KB -> ${new_kb}KB (resized 25%)"
        return
    fi

    # 4. Last resort: iterative 75% shrink until under limit
    local pass=1
    while [ "$(du -k "$f" | cut -f1)" -gt "$MAX_KB" ] && [ "$pass" -le 5 ]; do
        mogrify -strip -resize 75% -quality 80 "$f"
        pass=$((pass + 1))
    done
    new_kb=$(du -k "$f" | cut -f1)
    echo "  $f: ${orig_kb}KB -> ${new_kb}KB (iterative shrink)"
}
export -f compress_png

echo "=== Compressing PNGs in assets/ and docs/ (${JOBS} parallel jobs) ==="
oversized_count=$(find assets/ docs/ -name '*.png' -size +${MAX_KB}k 2>/dev/null | wc -l)
echo "  Found $oversized_count PNGs over ${MAX_KB}KB"

find assets/ docs/ -name '*.png' -size +${MAX_KB}k -print0 \
    | xargs -0 -P "$JOBS" -I {} bash -c 'compress_png "$@"' _ {} "$MAX_KB"

echo "=== Checking for remaining oversized PNGs ==="
oversized=$(find assets/ docs/ -name '*.png' -size +${MAX_KB}k 2>/dev/null)
if [ -n "$oversized" ]; then
    echo "WARNING: These PNGs still exceed ${MAX_KB}KB:"
    echo "$oversized" | while read -r f; do du -h "$f"; done
else
    echo "All PNGs are under ${MAX_KB}KB."
fi

echo "=== Done ==="
