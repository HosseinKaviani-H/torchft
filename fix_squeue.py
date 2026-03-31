import sys

f = "/home/dev/monarch/python/monarch/_src/job/slurm.py"
t = open(f).read()
old = "return jobs[0] if jobs else None"
new = """# SLURM 21.x returns multiple jobs; filter by ID
                for job in jobs:
                    if str(job.get("job_id")) == str(job_id):
                        return job
                return jobs[0] if jobs else None"""
if old not in t:
    print("Target line not found! Already patched?")
    sys.exit(1)
open(f, "w").write(t.replace(old, new, 1))
print("Done. Patched _get_job_info_json to filter by job_id.")
