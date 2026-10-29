"""Read-only host/GPU resource receipt; never modifies drivers or other jobs."""
from datetime import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]


def main():
    cpu=next((line.split(":",1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None)
    memory={line.split(':')[0]:line.split(':',1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in {'MemTotal','MemAvailable','SwapTotal','SwapFree'}}
    args=['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv']
    result=subprocess.run(args,capture_output=True,text=True,timeout=10)
    worker=json.loads((ROOT/'runs/worker.json').read_text()) if (ROOT/'runs/worker.json').exists() else {}
    proc={}
    status=Path('/proc')/str(worker.get('pid',-1))/'status'
    if status.exists():
        proc={line.split(':')[0]:line.split(':',1)[1].strip() for line in status.read_text().splitlines() if line.split(':')[0] in {'VmRSS','VmHWM','Threads'}}
    receipt={'captured_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'os':platform.freedesktop_os_release().get('PRETTY_NAME'),'cpu':cpu,'memory':memory,'project_disk':dict(zip(['total','used','free'],shutil.disk_usage(ROOT))),'gpu_query_status':'PASS' if result.returncode==0 else 'FAIL','gpu_csv':result.stdout.strip(),'worker_status':worker.get('status'),'worker_memory':proc,'scope':'Point-in-time read-only snapshot; stage GPU peaks are recorded in each experiment summary.'}
    path=ROOT/'logs/summary/resources.json';path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    main()
