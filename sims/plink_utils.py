"""
Helper functions for managing PLINK and external tools integration.
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, TextIO

def run_plink_conversion(
    vcf_writer: Callable[[TextIO], None],
    out_prefix: str,
    cm_map_path: str = None,
    threads: int | None = None,
    memory_mb: int | None = None,
) -> None:
    """
    Stream VCF text into PLINK 1.9 and write PLINK 1 binary format (.bed/.bim/.fam).

    If cm_map_path is provided (format: BP cM per line), updates the .bim file
    with correct genetic positions.
    """
    out_parent = Path(out_prefix).parent
    out_parent.mkdir(parents=True, exist_ok=True)

    plink_exe = shutil.which("plink")
    if plink_exe is None:
        raise FileNotFoundError("plink (PLINK 1.9) not found on PATH.")

    cmd = [
        plink_exe,
        "--vcf", "/dev/stdin",
        "--allow-extra-chr",
        "--biallelic-only", "strict",
        "--make-bed",
        "--out", out_prefix,
        "--silent",
    ]
    if threads is not None and int(threads) > 0:
        cmd.extend(["--threads", str(int(threads))])
    if memory_mb is not None and int(memory_mb) > 0:
        cmd.extend(["--memory", str(int(memory_mb))])
    print(f"Running PLINK conversion: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdin is not None
        vcf_writer(proc.stdin)
        proc.stdin.close()
    except Exception:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
        proc.kill()
        proc.wait()
        raise

    stderr = ""
    if proc.stderr is not None:
        stderr = proc.stderr.read()
        proc.stderr.close()
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"PLINK conversion failed:\n{stderr}")
    
    print(f"Created PLINK files: {out_prefix}.bed/bim/fam")
    
    # Inject Genetic Map if provided
    if cm_map_path and os.path.exists(cm_map_path):
        print(f"Injecting genetic map from {cm_map_path} into {out_prefix}.bim ...")
        
        # Load map: POS(int) -> cM(float)
        pos_to_cm = {}
        with open(cm_map_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    pos_to_cm[int(parts[0])] = parts[1] # Keep as string to preserve formatting if needed
        
        bim_path = f"{out_prefix}.bim"
        bim_tmp = f"{out_prefix}.bim.tmp"
        
        updated_count = 0
        try:
            with open(bim_path, "r") as fin, open(bim_tmp, "w") as fout:
                for line in fin:
                    # BIM format: CHR SNP CM BP A1 A2
                    cols = line.strip().split()
                    if len(cols) < 6:
                        fout.write(line)
                        continue
                        
                    bp = int(cols[3])
                    if bp in pos_to_cm:
                        cols[2] = pos_to_cm[bp]
                        updated_count += 1
                    
                    fout.write("\t".join(cols) + "\n")
            
            shutil.move(bim_tmp, bim_path)
            print(f"Updated {updated_count} variants with genetic positions.")

        except Exception as e:
            if os.path.exists(bim_tmp):
                os.remove(bim_tmp)
            raise RuntimeError(
                f"REQUIRED: Failed to update .bim file with genetic map: {e}. "
                f"Genetic positions are critical for LD-aware methods."
            )
