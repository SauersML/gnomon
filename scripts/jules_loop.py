"""
Jules Optimizer Loop - Continuous improvement cycle for Lean proofs.

This script runs within the 'jules_loop.yml' workflow.
It analyzes the build log from the current run (or a provided log file),
decides on improvements, sends a request to the Jules API, applies the patch,
and pushes the changes.

REGRESSION PROTECTION:
- If the Lean build was PASSING before Jules' changes, we verify it still passes
  after applying the patch. If the build now fails (regression), we abort.
- If the Lean build was already FAILING, we allow commits even if still failing,
  as any progress is better than no progress.
"""
import os
import sys
import json
import time
import subprocess
import requests
import re

JULES_API_URL = "https://jules.googleapis.com"
MAX_RETRIES = 2  # Number of retries for Jules API failures
RETRY_DELAY = 60  # Seconds between retries


def strip_ansi(text):
    """Removes ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def run_command(cmd, check=False):
    """Run a shell command and return stdout, stderr, return code."""
    if isinstance(cmd, list):
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    else:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def filter_noise(logs):
    """
    Remove noisy lines that don't help with debugging.
    These are mostly cache replay messages from Lake.
    """
    noise_patterns = [
        "Replayed Mathlib.",
        "Replayed Batteries.",
        "Replayed Qq.",
        "Replayed Aesop.",
        "Replayed ProofWidgets.",
        "Replayed LeanSearchClient.",
        "Replayed Plausible.",
        "Replayed ImportGraph.",
        "Replayed Cli.",
        "✔ [",  # Progress indicators like "✔ [649/918]"
        "Building ",  # "Building Mathlib.Order.Interval..."
        "Compiling ",  # "Compiling Mathlib..."
    ]
    
    filtered_lines = []
    for line in logs.split('\n'):
        # Skip lines that match any noise pattern
        is_noise = any(pattern in line for pattern in noise_patterns)
        if not is_noise:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def get_run_info():
    """
    Retrieves status and logs.
    First checks for local environment variables provided by the workflow.
    Falls back to fetching from GitHub API (legacy behavior, or for manual runs).
    """
    print("\n--- Getting Run Info ---")
    
    # Check for local log file and status provided by the calling workflow
    local_log_file = os.environ.get("LOCAL_LOG_FILE")
    local_status = os.environ.get("LOCAL_BUILD_STATUS")
    
    if local_log_file and local_status:
        print(f"Using local log file: {local_log_file} with status: {local_status}")
        try:
            with open(local_log_file, 'r') as f:
                raw_logs = f.read()

            # Filter noise first
            logs = filter_noise(strip_ansi(raw_logs))

            # Truncation logic: prioritize the END of the logs where errors usually are.
            max_len = 300000
            if len(logs) > max_len:
                print(f"Log size {len(logs)} exceeds {max_len}. Keeping last {max_len} characters.")
                logs = "..." + logs[-max_len:]

            return local_status, logs
        except Exception as e:
            print(f"Error reading local log file: {e}")
            return local_status, f"Error reading log file: {e}"

    # This might not work well if we are currently running inside the job we want logs for
    print("No local log file provided. This script is expected to run with LOCAL_LOG_FILE set.")
    return "unknown", "No logs available."


def verify_lean_build():
    """
    Run the Lean build and return True if it passes, False otherwise.
    """
    print("\n--- Verifying Lean Build ---")
    
    # Run the build
    stdout, stderr, code = run_command("lake build Calibrator")
    
    if code == 0:
        print("✅ Lean build passed.")
        return True
    else:
        print("❌ Lean build failed.")
        print(f"stderr: {stderr[:2000]}" if stderr else "")
        return False


def call_jules(prompt, attempt=1):
    """
    Interacts with Jules API to get a plan and changeset.
    Returns the changeset or None if Jules couldn't produce one.
    """
    api_key = os.environ.get("JULES_API_KEY")
    repo = os.environ.get("GITHUB_REPOSITORY")

    if not api_key:
        print("Error: JULES_API_KEY not set.")
        sys.exit(1)

    print(f"\n--- Initializing Jules Session (Attempt {attempt}/{MAX_RETRIES}) ---")

    payload = {
        "prompt": prompt,
        "sourceContext": {
            "source": f"sources/github/{repo}",
            "githubRepoContext": {"startingBranch": "main"}
        },
        "automationMode": "AUTO_CREATE_PR"
    }

    print("Sending payload to Jules API:")
    print(json.dumps(payload, indent=2))

    try:
        resp = requests.post(
            f"{JULES_API_URL}/v1alpha/sessions",
            headers={"X-Goog-Api-Key": api_key},
            json=payload,
            timeout=60
        )
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    if resp.status_code != 200:
        print(f"Failed to create session: {resp.text}")
        return None

    session = resp.json()
    session_name = session["name"]
    print(f"Session created: {session_name}")

    max_polls = 180  # 30 minutes of polling
    seen_ids = set()

    for i in range(max_polls):
        time.sleep(10)
        print(f"Polling activities... (Poll {i+1}/{max_polls})")

        try:
            r = requests.get(
                f"{JULES_API_URL}/v1alpha/{session_name}/activities",
                headers={"X-Goog-Api-Key": api_key},
                timeout=30
            )
        except requests.exceptions.RequestException as e:
            print(f"Polling error: {e}")
            continue

        if r.status_code != 200:
            print(f"Error polling: {r.text}")
            continue

        activities = r.json().get("activities", [])
        activities.sort(key=lambda x: x.get("createTime", ""))

        latest_changeset = None

        for act in activities:
            act_id = act.get("id")
            if act_id in seen_ids:
                continue
            seen_ids.add(act_id)

            originator = act.get("originator", "UNKNOWN")
            print(f"\n--- New Activity ({originator}) ---")

            if "planGenerated" in act:
                print("Plan Generated:")
                steps = act["planGenerated"].get("plan", {}).get("steps", [])
                for step in steps:
                    print(f"  {step.get('index', '?')}. {step.get('title', '')}")

            if "progressUpdated" in act:
                prog = act["progressUpdated"]
                print(f"Status: {prog.get('title', '')}")
                if "description" in prog:
                    print(f"Details: {prog['description']}")

            if "artifacts" in act:
                for art in act["artifacts"]:
                    if "bashOutput" in art:
                        bo = art["bashOutput"]
                        print(f"Bash Command: {bo.get('command')}")
                        print(f"Output:\n{bo.get('output')}")
                    if "changeSet" in art:
                        print("Artifact: ChangeSet found.")
                        latest_changeset = art["changeSet"]
                    if "pullRequest" in art:
                        pr = art["pullRequest"]
                        pr_url = pr.get('url', '')
                        print(f"Pull Request: {pr.get('title')} - {pr_url}")

                        # Add 'jules-loop' label to identify this as a Jules Loop PR
                        if pr_url:
                            try:
                                # Extract PR number from URL (e.g., https://github.com/owner/repo/pull/123)
                                pr_number = pr_url.rstrip('/').split('/')[-1]
                                print(f"Adding 'jules-loop' label to PR #{pr_number}...")

                                # Authenticate gh CLI with GITHUB_TOKEN
                                github_token = os.environ.get("GITHUB_TOKEN")
                                if github_token:
                                    subprocess.run(
                                        ['gh', 'auth', 'login', '--with-token'],
                                        input=github_token,
                                        check=True,
                                        capture_output=True,
                                        text=True
                                    )

                                # Use gh CLI to add label
                                result = subprocess.run(
                                    ['gh', 'pr', 'edit', pr_number, '--add-label', 'jules-loop'],
                                    check=True,
                                    capture_output=True,
                                    text=True
                                )
                                print("✓ Label added successfully")
                            except subprocess.CalledProcessError as e:
                                print(f"Warning: Failed to add label to PR: {e}")
                                print(f"stdout: {e.stdout}")
                                print(f"stderr: {e.stderr}")
                            except Exception as e:
                                print(f"Warning: Failed to add label to PR: {e}")

                        # If Jules created a PR directly, we're done
                        return "PR_CREATED"

            if "sessionCompleted" in act:
                print("Session Completed.")
                # Session completed - return whatever we have (or None if no changeset)
                if latest_changeset:
                    return latest_changeset
                print("Session completed but no ChangeSet was produced.")
                return None

        if latest_changeset:
            return latest_changeset

    print("Timed out waiting for Jules to produce a ChangeSet.")
    return None


def validate_patch(patch):
    """
    Validates the patch to ensure it doesn't:
    1. Create new files
    2. Delete theorems
    3. Only delete lines (zero additions)

    Returns (is_valid, reason) tuple.
    """
    print("\n--- Validating Patch ---")

    # Check if patch only has deletions (no additions)
    additions = 0
    deletions = 0
    deleted_theorem_lines = []
    new_files = []

    lines = patch.split('\n')
    for i, line in enumerate(lines):
        # Detect new file creation (diff shows "new file mode" or "/dev/null" in --- line)
        if line.startswith('--- /dev/null'):
            # Next line should be +++ b/filename
            if i + 1 < len(lines) and lines[i + 1].startswith('+++ b/'):
                filename = lines[i + 1][6:]  # Remove '+++ b/'
                new_files.append(filename)

        # Count additions and deletions
        if line.startswith('+') and not line.startswith('+++'):
            additions += 1
        elif line.startswith('-') and not line.startswith('---'):
            deletions += 1
            # Check if a theorem is being deleted
            if 'theorem ' in line:
                deleted_theorem_lines.append(line)

    print(f"Patch stats: +{additions} -{deletions}")

    # Rule 0: Reject if creating new files
    if new_files:
        print(f"\n🚫 REJECTED: Patch creates {len(new_files)} new file(s):")
        for filename in new_files:
            print(f"  {filename}")
        return False, "Patch creates new files (not allowed)"

    # Rule 1: Reject if deleting theorems
    if deleted_theorem_lines:
        print(f"\n🚫 REJECTED: Patch deletes {len(deleted_theorem_lines)} theorem(s):")
        for line in deleted_theorem_lines[:5]:  # Show first 5
            print(f"  {line[:100]}")
        if len(deleted_theorem_lines) > 5:
            print(f"  ... and {len(deleted_theorem_lines) - 5} more")
        return False, "Patch deletes theorems"

    # Rule 2: Reject if only deletions (no additions)
    if deletions > 0 and additions == 0:
        print(f"\n🚫 REJECTED: Patch only deletes lines (0 additions, {deletions} deletions)")
        return False, "Patch only deletes lines with zero additions"

    print("✅ Patch validation passed")
    return True, ""


def main():
    conclusion, logs = get_run_info()

    # Common restrictions for all prompts
    version_restriction = (
        "\n\nNOTE:\n"
        "- You are encouraged to proactively search the web.\n"
        "- DO NOT modify 'lean-toolchain' - the Lean version is intentionally pinned.\n"
        "- DO NOT modify version specifiers in 'lakefile.lean' (e.g., mathlib version).\n"
        "- Focus ONLY on proofs/*.lean files for improvements.\n"
        "- Always try to improve something--commit and finish. No further instruction will be given.\n"
        "- CRITICAL: DO NOT create new files. ONLY edit existing files.\n"
        "- CRITICAL: DO NOT delete theorems. DO NOT submit patches that only delete lines.\n"
    )

    # Build the prompt based on previous run status
    if conclusion == "success":
        prompt = (
            "The Lean Proof build passed successfully. "
            "Please find one thing to do or strengthen in the Lean proof files "
            "(specifically files in proofs/). Do not create new files. Do not edit any file besides 'proofs/Calibrator.lean'. You must successfully compile the code yourself. If the build times out or fails, do not submit it and keep working. You are not allowed to use 'native_decide' or similar."
            "If the build executes and terminates the shell, count that as a failure. Always tail build logs. You can optimize code, strengthen proofs, replace 'sorry' or 'axiom' with actual proofs. Feel free to try big or multiple tasks. We should also remove and fix specification gaming or vacuous verification. It involves writing theorem statements that appear rigorous in natural language but are mathematically constructed to be trivial or tautological. The most common tactic is begging the question, where the theorem explicitly includes the desired conclusion within its own hypothesis, rendering the proof a simple restatement of the input. Another tactic is the trivial witness, where a property regarding a complex mathematical object is proven by providing a hardcoded constant that technically satisfies a loose inequality without actually computing or representing the complex object itself. Finally, ex post facto construction involves defining a bounding function or rule only after calculating the specific error value, ensuring the condition is met by definition rather than by deriving a meaningful general law. For all of these, if they occur, we need to address them well and improve the code."
            "IMPORTANT: Ensure your changes compile and that all proofs are valid. Axioms are just as bad as sorrys and all axioms must be replaced with real proofs. Do not assume more than is necessary. Do not attempt low-importance small changes like style improvements, comments, etc."
            "Do not break existing functionality. DO NOT DELETE THEOREMS. DO NOT submit patches that only delete lines."
            + version_restriction
        )
    else:
        # Failure case
        prompt = (
            f"The Lean Proof build failed. "
            f"Here are the logs from the run (ANSI colors stripped):\n\n{logs}\n\n"
            "Please analyze the logs and fix the errors in the Lean proof files. If the code does not compile, you can commit a small improvement even if it is not a complete fix. You can search the web to find the latest documentation for the dependencies/libraries you're using. You can proactively find examples or code snippets that can help inform your edits. It's a good idea to web search."
            "You should check if your changes compile and that all proofs are valid. However, if the code does not compile, improve what you can as much as possible before submitting. It's okay if it still fails to compile as long as it is in a better state. At the same time, feel free to fix multiple issues at once, and don't afraid to make big improvements. You can do it!"
            + version_restriction
        )

    print(f"\nPrompting Jules with:\n{prompt}\n")

    # Retry loop for Jules API
    changeset = None
    for attempt in range(1, MAX_RETRIES + 1):
        changeset = call_jules(prompt, attempt)
        if changeset:
            break
        if attempt < MAX_RETRIES:
            print(f"\nJules didn't produce a changeset. Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    if not changeset:
        print("\nJules failed to produce a changeset after all retries.")
        sys.exit(0)

    # If Jules created a PR directly, we're done
    if changeset == "PR_CREATED":
        print("\nJules created a PR directly. Nothing more to do.")
        sys.exit(0)

    patch = changeset.get("gitPatch", {}).get("unidiffPatch")
    msg = changeset.get("gitPatch", {}).get("suggestedCommitMessage", "Jules Improvement")

    if not patch:
        print("\nJules returned a ChangeSet but no unidiffPatch.")
        sys.exit(0)

    print("\n--- Applying Patch ---")
    print(f"Patch content:\n{patch}\n")

    # Validate patch before applying
    is_valid, rejection_reason = validate_patch(patch)
    if not is_valid:
        print(f"\n🚫 PATCH REJECTED: {rejection_reason}")
        print("Aborting - no changes will be committed.")
        sys.exit(0)

    with open("jules.patch", "w") as f:
        f.write(patch)

    # Create a branch for the PR
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"jules-improvement-{timestamp}"
    
    run_command("git fetch origin main")
    run_command(f"git checkout -B {branch_name} origin/main")

    out, err, code = run_command("git apply jules.patch")
    if code != 0:
        print(f"Failed to apply patch: {err}")
        print("Patch may be malformed.")
        sys.exit(0)

    run_command('git config user.name "Jules Bot"')
    run_command('git config user.email "jules-bot@google.com"')

    run_command("git add .")
    _, _, code = run_command("git diff --cached --quiet")

    if code == 0:
        print("\nNo changes to commit after applying patch.")
        sys.exit(0)

    # --- Regression Check ---
    # If build was passing before (conclusion == "success"), verify it still passes.
    # If build was already failing, we allow commits even if still failing (progress is progress).
    was_passing_before = (conclusion == "success")
    
    if was_passing_before:
        print("\n--- Regression Check (build was passing before) ---")
        build_passes_now = verify_lean_build()
        
        if not build_passes_now:
            print("\n🚫 REGRESSION DETECTED: Build was passing, but Jules' changes broke it!")
            print("Reverting changes and aborting commit.")
            run_command("git checkout -- .")
            run_command("git clean -fd")
            sys.exit(0)
        else:
            print("✅ Regression check passed: build still works.")
    else:
        print("\n--- Skipping regression check (build was already failing) ---")
        print("Jules' changes will be committed even if build still fails.")
        # Optionally verify to log current state
        verify_lean_build()

    print("\n--- Committing and Creating PR ---")
    print(f"Commit message: {msg}")

    run_command(['git', 'commit', '-m', msg], check=True)

    print(f"Pushing branch {branch_name}...")
    run_command(f"git push origin {branch_name}", check=True)

    # Create PR using gh CLI
    print("Creating PR...")
    pr_title = msg.split('\n')[0][:80]  # First line of commit message, truncated
    pr_body = f"Automated improvement by Jules Loop.\n\nCommit message:\n```\n{msg}\n```"
    
    # Authenticate gh CLI
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        subprocess.run(
            ['gh', 'auth', 'login', '--with-token'],
            input=github_token,
            check=True,
            capture_output=True,
            text=True
        )
    
    out, err, code = run_command([
        'gh', 'pr', 'create',
        '--title', pr_title,
        '--body', pr_body,
        '--base', 'main',
        '--head', branch_name,
        '--label', 'jules-loop'
    ])
    
    if code != 0:
        print(f"Failed to create PR: {err}")
        sys.exit(1)
    
    print(f"✅ PR created: {out}")
    print("Done. PR will be validated by jules-pr-validator workflow.")


if __name__ == "__main__":
    main()
