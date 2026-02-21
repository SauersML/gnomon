"""
Helper functions for managing PLINK and external tools integration.
"""
import subprocess
import os
import shutil
import time
from pathlib import Path

def run_plink_conversion(
    vcf_path: str,
    out_prefix: str,
    cm_map_path: str = None,
    threads: int | None = None,
    memory_mb: int | None = None,
) -> None:
    """
    Convert VCF to PLINK binary format (.bed/.bim/.fam).
    Uses plink2.
    
    If cm_map_path is provided (format: BP cM per line), updates the .bim file
    with correct genetic positions.
    """
    out_parent = Path(out_prefix).parent
    out_parent.mkdir(parents=True, exist_ok=True)

    plink_exe = shutil.which("plink2")
    if plink_exe is None:
        raise FileNotFoundError("plink2 not found on PATH.")

    def _wait_for_file(path: str, timeout_s: float = 10.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if os.path.exists(path):
                return True
            time.sleep(0.1)
        return os.path.exists(path)
    
    def _run_plink(vcf_input: str) -> subprocess.CompletedProcess:
        cmd = [
            plink_exe,
            "--vcf", vcf_input,
            "--max-alleles", "2",
            "--allow-extra-chr",
            "--rm-dup", "exclude-all",
            "--make-bed",
            "--out", out_prefix,
            "--silent"
        ]
        if threads is not None and int(threads) > 0:
            cmd.extend(["--threads", str(int(threads))])
        if memory_mb is not None and int(memory_mb) > 0:
            cmd.extend(["--memory", str(int(memory_mb))])
        print(f"Running PLINK conversion: {' '.join(cmd)}")
        return subprocess.run(cmd, capture_output=True, text=True)

    # Wait briefly for potentially network-backed VCF metadata to become visible.
    if not _wait_for_file(vcf_path, timeout_s=10.0):
        raise RuntimeError(f"VCF input not found for PLINK conversion: {vcf_path}")

    result = _run_plink(vcf_path)

    if result.returncode != 0:
        raise RuntimeError(f"PLINK conversion failed:\n{result.stderr}")
        
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
