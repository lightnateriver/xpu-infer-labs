"""Build only the standalone CPU integer converter in this experiment directory."""
import hashlib,json,os,platform,shutil,subprocess
from pathlib import Path

root=Path(__file__).resolve().parent;compiler=shutil.which('g++');assert compiler,'g++ is required for the optional fast RGB converter'
target=root/'portable_rgb_core.so';assert not target.exists(),'Keep existing build receipts immutable'
command=[compiler,'-O3','-std=c++17','-fPIC','-shared',str(root/'portable_rgb_core.cpp'),'-o',str(target)]
temporary=root/'compiler-tmp';temporary.mkdir(exist_ok=True)
environment=dict(os.environ,TMPDIR=str(temporary))
process=subprocess.run(command,env=environment,text=True,capture_output=True)
if process.returncode:
    (root/'rgb-core-build-failure.json').write_text(json.dumps({'command':command,'exit_code':process.returncode,'stdout':process.stdout,'stderr':process.stderr},indent=2))
    raise RuntimeError('CPU RGB converter build failed; see preserved failure receipt')
receipt={'machine':platform.machine(),'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],'command':command,'source_sha256':hashlib.sha256((root/'portable_rgb_core.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
(root/'rgb-core-build.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
