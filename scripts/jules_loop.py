#!/usr/bin/env python3
"""
Jules Optimizer Loop - Infinite improvement cycle for Lean proofs.

This script is part of an infinite loop:
  prover.yml (validates Lean) → jules_loop.yml (improves Lean) → push → repeat

The loop should NEVER stop except for API rate limits. All other failures trigger retries.
"""
import os
import sys
import json
import time
import subprocess
import requests
import re

JULES_API_URL = "https://jules.googleapis.com"
MAX_RETRIES = 3  # Number of retries for Jules API failures
RETRY_DELAY = 30  # Seconds between retries


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


def fetch_logs(run_id):
    """
    Fetches logs for a specific workflow run.
    Filters out noisy cache replay lines and prioritizes error information.
    """
    print(f"Fetching logs for run {run_id}...")
    out, err, code = run_command(f"gh run view {run_id} --log")
    if code != 0:
        print(f"Failed to fetch logs: {err}")
        return f"Failed to fetch logs: {err}"
    
    raw_logs = strip_ansi(out)
    print(f"Raw logs fetched. Length: {len(raw_logs)} characters.")
    
    # Filter out noisy cache replay lines
    filtered_logs = filter_noise(raw_logs)
    print(f"After filtering noise: {len(filtered_logs)} characters.")
    
    # Truncate to 300k chars if still too long
    if len(filtered_logs) > 300000:
        print("Truncating logs to 300,000 characters...")
        filtered_logs = filtered_logs[:300000]
    
    return filtered_logs


def get_previous_run_info():
    """Retrieves status and logs from the workflow run that triggered this script."""
    print("\n--- Getting Previous Run Info ---")
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    # Try to use the triggering workflow run info
    if event_path:
        print(f"Reading event from {event_path}")
        try:
            with open(event_path) as f:
                event = json.load(f)

            run = event.get("workflow_run")
            if run:
                conclusion = run.get("conclusion")
                run_id = run.get("id")
                workflow_id = run.get("workflow_id")
                print(f"Triggering run {run_id} found in event. Conclusion: {conclusion}")

                logs = ""
                if conclusion != "success":
                    logs = fetch_logs(run_id)
                else:
                    logs = "Build passed successfully."

                return conclusion, logs, workflow_id
            else:
                print("No 'workflow_run' found in event payload.")
        except Exception as e:
            print(f"Warning: Failed to read or parse GITHUB_EVENT_PATH: {e}")

    # Fallback: Fetch latest run of prover.yml via gh
    print("No triggering workflow_run found. Fetching latest 'Lean Prover CI' run...")

    out, err, code = run_command(
        ["gh", "run", "list", "--workflow", "prover.yml", "--limit", "1", 
         "--json", "conclusion,databaseId,status"]
    )

    if code != 0 or not out:
        print(f"Failed to fetch runs: {err}")
        return None, None, None

    try:
        runs = json.loads(out)
    except json.JSONDecodeError:
        print(f"Failed to decode gh output: {out}")
        return None, None, None

    if not runs:
        print("No runs found for prover.yml")
        return None, None, None

    latest_run = runs[0]
    conclusion = latest_run.get("conclusion")
    run_id = latest_run.get("databaseId")
    workflow_id = "prover.yml"

    print(f"Latest run {run_id} status: {latest_run.get('status')}, conclusion: {conclusion}")

    logs = ""
    if conclusion != "success":
        logs = fetch_logs(run_id)
    else:
        logs = "Build passed successfully."

    return conclusion, logs, workflow_id


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
        }
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

    max_polls = 60  # 10 minutes of polling
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
                        print(f"Pull Request: {pr.get('title')} - {pr.get('url')}")

            if "sessionCompleted" in act:
                print("Session Completed.")

        if latest_changeset:
            return latest_changeset

    print("Timed out waiting for Jules to produce a ChangeSet.")
    return None


def trigger_next_cycle(workflow_id):
    """Trigger the next prover.yml run to continue the loop."""
    target_workflow = workflow_id if workflow_id else "prover.yml"
    print(f"\n--- Triggering Next Workflow ({target_workflow}) ---")
    out, err, code = run_command(f"gh workflow run {target_workflow}")
    if code == 0:
        print("Successfully triggered next cycle.")
    else:
        print(f"Failed to trigger next cycle: {err}")


def main():
    conclusion, logs, workflow_id = get_previous_run_info()

    # Common restrictions for all prompts
    version_restriction = (
        "\n\nCRITICAL RESTRICTIONS:\n"
        "- DO NOT modify 'lean-toolchain' - the Lean version is intentionally pinned\n"
        "- DO NOT modify version specifiers in 'lakefile.lean' (e.g., mathlib version)\n"
        "- Focus ONLY on proofs/*.lean files for improvements\n"
    )

    # Build the prompt based on previous run status
    if conclusion == "success":
        prompt = (
            "The previous Lean Proof GHA run passed successfully. "
            "Please find one thing to do, improve, fix, or strengthen in the Lean proof files "
            "(specifically files in proofs/). "
            "You can optimize code, add comments, strengthen proofs, replace 'sorry' with actual proofs, or refactor. "
            "IMPORTANT: Ensure your changes compile and that all proofs are valid. "
            "Do not break existing functionality."
            + version_restriction
        )
    elif conclusion:
        prompt = (
            f"The previous Lean Proof GHA run failed. "
            f"Here are the logs from the run (ANSI colors stripped):\n\n{logs}\n\n"
            "Please analyze the logs and fix the errors in the Lean proof files. "
            "IMPORTANT: Ensure your changes compile and that all proofs are valid."
            + version_restriction
        )
    else:
        prompt = (
            "Please find one thing to improve, fix, or strengthen in the Lean proof files "
            "(specifically files in proofs/). "
            "IMPORTANT: Ensure your changes compile and that all proofs are valid."
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
        print("Triggering next cycle to keep the loop alive...")
        trigger_next_cycle(workflow_id)
        sys.exit(0)  # Exit gracefully, the triggered workflow continues the loop

    patch = changeset.get("gitPatch", {}).get("unidiffPatch")
    msg = changeset.get("gitPatch", {}).get("suggestedCommitMessage", "Jules Improvement")

    if not patch:
        print("\nJules returned a ChangeSet but no unidiffPatch.")
        print("Triggering next cycle to keep the loop alive...")
        trigger_next_cycle(workflow_id)
        sys.exit(0)

    print("\n--- Applying Patch ---")
    print(f"Patch content:\n{patch}\n")

    with open("jules.patch", "w") as f:
        f.write(patch)

    # Ensure we are on main branch
    run_command("git fetch origin main")
    run_command("git checkout -B main origin/main")

    out, err, code = run_command("git apply jules.patch")
    if code != 0:
        print(f"Failed to apply patch: {err}")
        print("Patch may be malformed. Triggering next cycle to keep the loop alive...")
        trigger_next_cycle(workflow_id)
        sys.exit(0)

    run_command('git config user.name "Jules Bot"')
    run_command('git config user.email "jules-bot@google.com"')

    run_command("git add .")
    _, _, code = run_command("git diff --cached --quiet")

    if code == 0:
        print("\nNo changes to commit after applying patch.")
        print("Triggering next cycle to keep the loop alive...")
        trigger_next_cycle(workflow_id)
        sys.exit(0)

    print("\n--- Committing and Pushing ---")
    print(f"Commit message: {msg}")

    run_command(['git', 'commit', '-m', msg], check=True)

    print("Pulling latest changes to avoid non-fast-forward...")
    run_command("git pull --rebase origin main", check=True)

    print("Pushing changes...")
    run_command("git push origin main", check=True)

    # Note: We don't need to trigger prover.yml manually here because
    # the push to main will automatically trigger it via the 'push' event.
    print("\nLoop iteration complete. Push will trigger prover.yml automatically.")


if __name__ == "__main__":
    main()
