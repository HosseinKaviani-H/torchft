import re, shutil
f = "/home/dev/monarch/python/monarch/_src/job/slurm.py"
shutil.copy(f, f + ".bak2")
lines = open(f).readlines()
for i, line in enumerate(lines):
    if "_kill() called for job" in line or "kill_debug" in line:
        indent = len(line) - len(line.lstrip())
        lines[i] = " " * indent + 'import traceback; open("/tmp/kill_debug.log","a").write("_kill called for job " + str(self._slurm_job_id) + "\\n" + "".join(traceback.format_stack()) + "\\n")\n'
        print(f"Replaced line {i+1}")
        break
else:
    print("Target line not found!")
open(f, "w").writelines(lines)
print("Done. Backup at " + f + ".bak2")
