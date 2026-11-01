import os
import sys
import subprocess
from pathlib import Path

def sdd_workspace(plan_path: str) -> Path:
    plan = Path(plan_path)
    if not plan.exists():
        raise FileNotFoundError(f"No such plan file: {plan_path}")
    slug = plan.stem
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    base = root / ".superpowers" / "sdd"
    dir_path = base / slug
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(base / ".gitignore", "w") as f:
        f.write("*\n")
    return dir_path

def task_brief(plan_path: str, task_num: int, out_file: str = None) -> Path:
    plan = Path(plan_path)
    if not plan.exists():
        raise FileNotFoundError(f"No such plan file: {plan_path}")
    
    if out_file:
        out_path = Path(out_file)
    else:
        ws = sdd_workspace(plan_path)
        out_path = ws / f"task-{task_num}-brief.md"
        
    lines = plan.read_text(encoding="utf-8").splitlines()
    task_lines = []
    in_task = False
    in_fence = False
    
    target_heading = f"Task {task_num}:"
    
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("### Task "):
            if target_heading in line:
                in_task = True
            elif in_task:
                # Reached next task
                break
        if in_task:
            task_lines.append(line)
            
    if not task_lines:
        raise ValueError(f"Task {task_num} not found in {plan_path}")
        
    out_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: {len(task_lines)} lines")
    return out_path

def review_package(plan_path: str, base_commit: str, head_commit: str, out_file: str = None) -> Path:
    plan = Path(plan_path)
    if not plan.exists():
        raise FileNotFoundError(f"No such plan file: {plan_path}")
        
    if out_file:
        out_path = Path(out_file)
    else:
        ws = sdd_workspace(plan_path)
        base_short = subprocess.check_output(["git", "rev-parse", "--short", base_commit], text=True).strip()
        head_short = subprocess.check_output(["git", "rev-parse", "--short", head_commit], text=True).strip()
        out_path = ws / f"review-{base_short}..{head_short}.diff"
        
    log_out = subprocess.check_output(["git", "log", "--oneline", f"{base_commit}..{head_commit}"], encoding="utf-8", errors="replace")
    stat_out = subprocess.check_output(["git", "diff", "--stat", f"{base_commit}..{head_commit}"], encoding="utf-8", errors="replace")
    diff_out = subprocess.check_output(["git", "diff", "-U10", f"{base_commit}..{head_commit}"], encoding="utf-8", errors="replace")
    
    content = f"# Review package: {base_commit}..{head_commit}\n\n## Commits\n{log_out}\n## Files changed\n{stat_out}\n## Diff\n{diff_out}\n"
    out_path.write_text(content, encoding="utf-8")
    print(f"wrote {out_path}: {len(diff_out)} bytes")
    return out_path

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "workspace":
        print(sdd_workspace(sys.argv[2]))
    elif cmd == "brief":
        print(task_brief(sys.argv[2], int(sys.argv[3])))
    elif cmd == "review":
        print(review_package(sys.argv[2], sys.argv[3], sys.argv[4]))
