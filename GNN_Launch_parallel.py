import argparse
import os
import sys
import time

from flyvis.utils.compute_cloud_utils import LSFManager, wait_for_many


def build_job(task: str, config: str, model_id: str, ensemble_id: str | None,
              python_exec: str, script_path: str,
              n_cpus: int, queue: str, gpu: str, out_dir: str) -> tuple[str, str]:
    cm = LSFManager()
    job_name = f"ng_{config}_mid{model_id}"
    outfile = os.path.join(out_dir, f"{job_name}_{int(time.time())}.out")
    submit = cm.get_submit_command(job_name=job_name, n_cpus=n_cpus, output_file=outfile, gpu=gpu, queue=queue)

    # Compose script invocation: use the new multimodel main but pin to a single model id
    opt = f"-o {task} {config} --model-start {int(model_id)} --model-end {int(model_id)}"
    if ensemble_id:
        opt += f" --ensemble-id {ensemble_id}"
    script_cmd = cm.get_script_part(f"{python_exec} {script_path} {opt}")
    return submit + script_cmd, job_name


def main():
    parser = argparse.ArgumentParser(description="Launch per-model NeuralGraph jobs in parallel (LSF)")
    parser.add_argument("--configs", nargs="+", required=True, help="One or more config names (e.g. fly_N9_22_10 fly_N9_44_24)")
    parser.add_argument("--task", default="generate,train,test", help="Task token containing any of generate,train,test")
    parser.add_argument("--model-start", type=int, default=0)
    parser.add_argument("--model-end", type=int, default=49)
    parser.add_argument("--ensemble-id", type=str, default=None)
    parser.add_argument("--queue", type=str, default="gpu_h100")
    parser.add_argument("--gpu", type=str, default="num=1")
    parser.add_argument("--ncpus", type=int, default=4)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--script", type=str, default=os.path.abspath("GNN_Main_multimodel.py"))
    parser.add_argument("--outdir", type=str, default=os.path.abspath("./job_logs"))
    parser.add_argument("--dry", action="store_true", default=False)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    job_id_names = {}
    cm = LSFManager()
    for cfg in args.configs:
        for mid in range(args.model_start, args.model_end + 1):
            cmd, name = build_job(
                task=args.task,
                config=cfg,
                model_id=f"{mid:03d}",
                ensemble_id=args.ensemble_id,
                python_exec=args.python,
                script_path=args.script,
                n_cpus=args.ncpus,
                queue=args.queue,
                gpu=args.gpu,
                out_dir=args.outdir,
            )
            if args.dry:
                print(f"DRY: {cmd}")
                job_id = f"dry_{name}_{mid:03d}"
            else:
                job_id = cm.run_job(cmd)
            job_id_names[job_id] = name

    wait_for_many(job_id_names, dry=args.dry)


if __name__ == "__main__":
    main()


