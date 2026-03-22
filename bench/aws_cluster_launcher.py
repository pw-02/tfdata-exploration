#!/usr/bin/env python3
import argparse
import json
import shlex
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import boto3


@dataclass
class NodeGroupSpec:
    name: str
    count: int
    instance_type: str
    ami_id: str
    subnet_id: str
    security_group_ids: List[str]
    iam_instance_profile_name: str
    key_name: Optional[str] = None
    user_data: Optional[str] = None


class AWSClusterLauncher:
    def __init__(self, region: str):
        self.region = region
        self.ec2 = boto3.client("ec2", region_name=region)
        self.ssm = boto3.client("ssm", region_name=region)

    def launch_instances(
        self,
        spec: NodeGroupSpec,
        cluster_name: str,
        role: str,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        tag_map = {
            "Name": f"{cluster_name}-{role}",
            "Cluster": cluster_name,
            "Role": role,
            "NodeGroup": spec.name,
            "ManagedBy": "aws_cluster_launcher",
        }

        if extra_tags:
            tag_map.update(extra_tags)

        tags = [{"Key": k, "Value": v} for k, v in tag_map.items()]

        params = {
            "ImageId": spec.ami_id,
            "InstanceType": spec.instance_type,
            "MinCount": spec.count,
            "MaxCount": spec.count,
            "SubnetId": spec.subnet_id,
            "SecurityGroupIds": spec.security_group_ids,
            "IamInstanceProfile": {"Name": spec.iam_instance_profile_name},
            "TagSpecifications": [
                {"ResourceType": "instance", "Tags": tags},
                {"ResourceType": "volume", "Tags": tags},
            ],
        }

        if spec.key_name:
            params["KeyName"] = spec.key_name
        if spec.user_data:
            params["UserData"] = spec.user_data

        resp = self.ec2.run_instances(**params)
        instance_ids = [i["InstanceId"] for i in resp["Instances"]]
        print(f"Launched {len(instance_ids)} {role} instances: {instance_ids}")
        return instance_ids



    def find_existing_instances(
        self,
        cluster_name: str,
        role: str,
        nodegroup_name: Optional[str] = None,
    ) -> List[Dict]:
        filters = [
            {"Name": "tag:Cluster", "Values": [cluster_name]},
            {"Name": "tag:Role", "Values": [role]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ]
        if nodegroup_name:
            filters.append({"Name": "tag:NodeGroup", "Values": [nodegroup_name]})

        paginator = self.ec2.get_paginator("describe_instances")
        out = []
        for page in paginator.paginate(Filters=filters):
            for reservation in page["Reservations"]:
                for inst in reservation["Instances"]:
                    out.append(inst)
        return out

    def _instance_matches_spec(self, inst: Dict, spec: NodeGroupSpec) -> bool:
        if inst.get("InstanceType") != spec.instance_type:
            return False

        if inst.get("ImageId") != spec.ami_id:
            return False

        if inst.get("SubnetId") != spec.subnet_id:
            return False

        actual_sgs = sorted(sg["GroupId"] for sg in inst.get("SecurityGroups", []))
        expected_sgs = sorted(spec.security_group_ids)
        if actual_sgs != expected_sgs:
            return False

        profile = inst.get("IamInstanceProfile", {})
        arn = profile.get("Arn", "")
        if spec.iam_instance_profile_name and not arn.endswith(
            f"instance-profile/{spec.iam_instance_profile_name}"
        ):
            return False

        if spec.key_name and inst.get("KeyName") != spec.key_name:
            return False

        return True

    def ensure_instances(
        self,
        spec: NodeGroupSpec,
        cluster_name: str,
        role: str,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        existing = self.find_existing_instances(
            cluster_name=cluster_name,
            role=role,
            nodegroup_name=spec.name,
        )

        matching = [inst for inst in existing if self._instance_matches_spec(inst, spec)]
        matching = sorted(matching, key=lambda x: x["InstanceId"])

        chosen = matching[:spec.count]
        chosen_ids = [inst["InstanceId"] for inst in chosen]

        stopped_ids = [inst["InstanceId"] for inst in chosen if inst["State"]["Name"] == "stopped"]
        if stopped_ids:
            self.ec2.start_instances(InstanceIds=stopped_ids)
            print(f"Started stopped {role} instances: {stopped_ids}")

        missing = spec.count - len(chosen_ids)
        if missing <= 0:
            print(f"Reusing existing {role} instances: {chosen_ids}")
            return chosen_ids

        launch_spec = NodeGroupSpec(
            name=spec.name,
            count=missing,
            instance_type=spec.instance_type,
            ami_id=spec.ami_id,
            subnet_id=spec.subnet_id,
            security_group_ids=spec.security_group_ids,
            iam_instance_profile_name=spec.iam_instance_profile_name,
            key_name=spec.key_name,
            user_data=spec.user_data,
        )

        merged_tags = {"NodeGroup": spec.name}
        if extra_tags:
            merged_tags.update(extra_tags)

        new_ids = self.launch_instances(
            spec=launch_spec,
            cluster_name=cluster_name,
            role=role,
            extra_tags=merged_tags,
        )
        return chosen_ids + new_ids

    def wait_for_instances_running(self, instance_ids: List[str], timeout_sec: int = 900) -> None:
        if not instance_ids:
            return
        waiter = self.ec2.get_waiter("instance_running")
        waiter.wait(
            InstanceIds=instance_ids,
            WaiterConfig={"Delay": 10, "MaxAttempts": max(1, timeout_sec // 10)},
        )
        print(f"Instances are running: {instance_ids}")

    def describe_instances(self, instance_ids: List[str]) -> List[Dict]:
        if not instance_ids:
            return []
        resp = self.ec2.describe_instances(InstanceIds=instance_ids)
        out = []
        for r in resp["Reservations"]:
            for inst in r["Instances"]:
                out.append(inst)
        return out

    def get_private_ip_map(self, instance_ids: List[str]) -> Dict[str, str]:
        details = self.describe_instances(instance_ids)
        return {inst["InstanceId"]: inst.get("PrivateIpAddress", "") for inst in details}

    def wait_for_ssm_online(self, instance_ids: List[str], timeout_sec: int = 900) -> Dict[str, str]:
        if not instance_ids:
            return {}

        deadline = time.time() + timeout_sec
        found = {}

        while time.time() < deadline:
            resp = self.ssm.describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": instance_ids}]
            )
            for item in resp.get("InstanceInformationList", []):
                if item.get("PingStatus") == "Online":
                    found[item["InstanceId"]] = item["InstanceId"]

            if len(found) == len(instance_ids):
                print(f"All instances are online in SSM: {instance_ids}")
                return found

            print(f"Waiting for SSM online: {len(found)}/{len(instance_ids)}")
            time.sleep(10)

        missing = sorted(set(instance_ids) - set(found.keys()))
        raise TimeoutError(f"Timed out waiting for SSM online. Missing: {missing}")

    def send_shell_commands(
        self,
        instance_ids: List[str],
        commands: List[str],
        comment: str,
        timeout_sec: int = 3600,
    ) -> str:
        resp = self.ssm.send_command(
            InstanceIds=instance_ids,
            DocumentName="AWS-RunShellScript",
            Comment=comment,
            Parameters={
                "commands": commands,
                "executionTimeout": [str(timeout_sec)],
            },
            CloudWatchOutputConfig={"CloudWatchOutputEnabled": False},
        )
        cmd_id = resp["Command"]["CommandId"]
        print(f"Sent SSM command {cmd_id} to {instance_ids}: {comment}")
        return cmd_id
    
    def wait_for_command(
        self,
        command_id: str,
        instance_ids: List[str],
        timeout_sec: int = 3600,
    ) -> None:
        deadline = time.time() + timeout_sec
        done = set()

        while time.time() < deadline:
            all_ok = True

            for iid in instance_ids:
                if iid in done:
                    continue

                try:
                    inv = self.ssm.get_command_invocation(
                        CommandId=command_id,
                        InstanceId=iid,
                    )
                except self.ssm.exceptions.InvocationDoesNotExist:
                    # SSM Run Command is eventually consistent.
                    all_ok = False
                    continue

                status = inv["Status"]

                if status == "Success":
                    done.add(iid)
                elif status in {"Pending", "InProgress", "Delayed", "Cancelling"}:
                    all_ok = False
                else:
                    raise RuntimeError(
                        f"Command {command_id} failed on {iid} with status={status}\n"
                        f"stdout:\n{inv.get('StandardOutputContent', '')}\n"
                        f"stderr:\n{inv.get('StandardErrorContent', '')}"
                    )

            if all_ok and len(done) == len(instance_ids):
                print(f"Command {command_id} completed successfully on all targets.")
                return

            time.sleep(5)

        raise TimeoutError(f"Timed out waiting for command {command_id}")
    
    def stop_instances(self, instance_ids: List[str]) -> None:
        if not instance_ids:
            return
        self.ec2.stop_instances(InstanceIds=instance_ids)
        print(f"Stop requested for: {instance_ids}")
    
    def terminate_instances(self, instance_ids: List[str]) -> None:
        if not instance_ids:
            return
        self.ec2.terminate_instances(InstanceIds=instance_ids)
        print(f"Terminate requested for: {instance_ids}")

def _bootstrap_repo_commands(code_dir: str) -> List[str]:
    repo_url = "https://github.com/pw-02/tfdata-exploration.git"
    repo_parent = "/".join(code_dir.rstrip("/").split("/")[:-1]) or "/"
    repo_name = code_dir.rstrip("/").split("/")[-1]
    venv_dir = f"{code_dir}/.venv"

    return [
        "set -eux",

        # 🔥 Kill old cluster processes (safe, targeted)
        "pkill -f dispatcher.py || true",
        "pkill -f worker.py || true",
        "pkill -f run_benchmark.py || true",
        "pkill -f trainer.py || true",

        "sleep 2",

        (
            "SUDO='' ; "
            "if command -v sudo >/dev/null 2>&1; then SUDO='sudo'; fi ; "
            "if command -v apt-get >/dev/null 2>&1; then "
            "  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -y; "
            "  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "    -o Dpkg::Use-Pty=0 "
            "    -o Dpkg::Options::=--force-confdef "
            "    -o Dpkg::Options::=--force-confold "
            "    git python3 python3-pip python3-venv; "
            "elif command -v dnf >/dev/null 2>&1; then "
            "  $SUDO dnf install -y git python3 python3-pip; "
            "elif command -v yum >/dev/null 2>&1; then "
            "  $SUDO yum install -y git python3 python3-pip; "
            "else "
            "  echo 'No supported package manager found'; exit 1; "
            "fi"
        ),
        f"mkdir -p {repo_parent}",
        f"cd {repo_parent}",
        (
            f"if [ ! -d {repo_name} ]; then "
            f"git clone {repo_url} {repo_name}; "
            "fi"
        ),
        f"cd {code_dir}",
        "git pull --ff-only || true",
        f"python3 -m venv {venv_dir}",
        f". {venv_dir}/bin/activate",
        "python -m pip install --upgrade pip",
        "python -m pip install -r requirements.txt",
        "mkdir -p logs",
    ]

def build_dispatcher_command(
    repo_dir: str,
    run_dir: str,
    dispatcher_port: int,
    work_dir: str,
) -> List[str]:
    cmds = _bootstrap_repo_commands(repo_dir)
    venv_activate = f". {repo_dir}/.venv/bin/activate"

    cmds.extend([
        venv_activate,
        f"cd {run_dir}",
        "pwd",
        "ls -la",
        "mkdir -p logs",
        f"mkdir -p {work_dir}",
        (
            "if pgrep -af 'dispatcher.py' >/dev/null; then "
            "echo 'dispatcher already running'; "
            "else "
            f"nohup python -u dispatcher.py "
            f"--port {dispatcher_port} "
            f"--work-dir {work_dir} "
            f"> logs/dispatcher.log 2>&1 & "
            "fi"
        ),
        "sleep 5",
        (
            "if ! pgrep -af 'dispatcher.py' >/dev/null; then "
            "echo 'dispatcher failed to start'; "
            "echo '==== dispatcher.log ===='; "
            "cat logs/dispatcher.log || true; "
            "echo '==== end dispatcher.log ===='; "
            "exit 1; "
            "fi"
        ),
    ])
    return cmds


def build_worker_commands(
    repo_dir: str,
    run_dir: str,
    dispatcher_address: str,
    worker_ports: List[int],
    worker_host: str,
) -> List[str]:
    cmds = _bootstrap_repo_commands(repo_dir)
    venv_activate = f". {repo_dir}/.venv/bin/activate"

    cmds.extend([
        venv_activate,
        f"cd {run_dir}",
        "pwd",
        "ls -la",
        "mkdir -p logs",
    ])

    for port in worker_ports:
        cmds.append(
            (
                f"if pgrep -af 'worker.py.*--port {port}\\b' >/dev/null; then "
                f"echo 'worker {port} already running'; "
                "else "
                f"nohup python -u worker.py "
                f"--dispatcher-address {dispatcher_address} "
                f"--port {port} "
                f"--worker-address {worker_host}:{port} "
                f"> logs/worker_{port}.log 2>&1 & "
                "fi"
            )
        )
        cmds.append("sleep 3")
        cmds.append(
            (
                f"if ! pgrep -af 'worker.py.*--port {port}\\b' >/dev/null; then "
                f"echo 'worker {port} failed to start'; "
                f"echo '==== worker_{port}.log ===='; "
                f"cat logs/worker_{port}.log || true; "
                f"echo '==== end worker_{port}.log ===='; "
                "exit 1; "
                "fi"
            )
        )

    return cmds

def build_trainer_command(
    repo_dir: str,
    run_dir: str,
    service: Optional[str],
    mode: str,
    model: str,
    path: str,
    steps: int,
    batch_size: int,
    num_trainers: int,
    repeat: bool,
    shuffle: bool,
    trainer_stagger_sec: float,
    out_dir: str,
) -> List[str]:
    cmds = _bootstrap_repo_commands(repo_dir)

    venv_activate = f". {shlex.quote(repo_dir)}/.venv/bin/activate"
    repeat_flag = "--repeat" if repeat else ""
    shuffle_flag = "--shuffle" if shuffle else ""
    service_flag = f"--service {shlex.quote(service)}" if service else ""

    cmds.extend([
        venv_activate,
        f"cd {shlex.quote(run_dir)}",
        "pwd",
        "ls -la",
        f"mkdir -p {shlex.quote(out_dir)}",
        (
            f"python -u run_benchmark.py "
            f"--mode {shlex.quote(mode)} "
            f"{service_flag} "
            f"--model {shlex.quote(model)} "
            f"--path {shlex.quote(path)} "
            f"--steps {steps} "
            f"--batch-size {batch_size} "
            f"--num-trainers {num_trainers} "
            f"--trainer-stagger-sec {trainer_stagger_sec} "
            f"--out-dir {shlex.quote(out_dir)} "
            f"{repeat_flag} {shuffle_flag}"
        ),
        f"ls -R {shlex.quote(out_dir)} || true",
    ])
    return cmds

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="bench/cluster_config.json",
        help="Path to cluster config JSON",
    )
    parser.add_argument(
        "--terminate-on-finish",
        action="store_true",
        default=False,
        help="Terminate cluster instances when benchmark finishes",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    region = cfg["region"]
    cluster_name = cfg["cluster_name"]

    launcher = AWSClusterLauncher(region=region)

    dispatcher_spec = NodeGroupSpec(**cfg["dispatcher_node"])
    worker_spec = NodeGroupSpec(**cfg["worker_node"])
    trainer_spec = NodeGroupSpec(**cfg["trainer_node"])

    code_dir = cfg["code_dir"]
    dispatcher_port = int(cfg.get("dispatcher_port", 5000))
    dispatcher_work_dir = cfg.get("dispatcher_work_dir", "/tmp/tf_data_dispatcher")
    worker_ports_per_node = cfg.get("worker_ports_per_node", [5001])
    benchmark = cfg["benchmark"]

    launched = {"dispatcher": [], "workers": [], "trainers": []}

    try:
        # 1) Ensure dispatcher
        dispatcher_ids = launcher.ensure_instances(
            spec=dispatcher_spec,
            cluster_name=cluster_name,
            role="dispatcher",
        )
        launched["dispatcher"] = dispatcher_ids
        launcher.wait_for_instances_running(dispatcher_ids)
        launcher.wait_for_ssm_online(dispatcher_ids)
        dispatcher_ip = launcher.get_private_ip_map(dispatcher_ids)[dispatcher_ids[0]]

        # 2) Ensure dispatcher process
        cmd_id = launcher.send_shell_commands(
            instance_ids=dispatcher_ids,
            commands=build_dispatcher_command(
                repo_dir=code_dir,
                dispatcher_port=dispatcher_port,
                work_dir=dispatcher_work_dir,
                run_dir=cfg["run_dir"],
            ),
            comment="Start tf.data dispatcher if needed",
        )
        launcher.wait_for_command(cmd_id, dispatcher_ids)

        dispatcher_address = f"{dispatcher_ip}:{dispatcher_port}"
        service = f"grpc://{dispatcher_address}"
        print(f"Dispatcher service: {service}")

        # 3) Ensure worker nodes
        worker_ids = launcher.ensure_instances(
            spec=worker_spec,
            cluster_name=cluster_name,
            role="worker",
        )
        launched["workers"] = worker_ids
        launcher.wait_for_instances_running(worker_ids)
        launcher.wait_for_ssm_online(worker_ids)
        worker_ip_map = launcher.get_private_ip_map(worker_ids)

        # 4) Ensure worker processes
        for iid in worker_ids:
            worker_host = worker_ip_map[iid]
            cmd_id = launcher.send_shell_commands(
                instance_ids=[iid],
                commands=build_worker_commands(
                    repo_dir=code_dir,
                    dispatcher_address=dispatcher_address,
                    worker_ports=worker_ports_per_node,
                    worker_host=worker_host,
                    run_dir=cfg["run_dir"],
                ),
                comment=f"Start tf.data workers on {iid} if needed",
            )
            launcher.wait_for_command(cmd_id, [iid])

        # 5) Ensure trainer nodes
        trainer_ids = launcher.ensure_instances(
            spec=trainer_spec,
            cluster_name=cluster_name,
            role="trainer",
        )
        launched["trainers"] = trainer_ids
        launcher.wait_for_instances_running(trainer_ids)
        launcher.wait_for_ssm_online(trainer_ids)

        # 6) Run benchmark on trainer nodes
        for idx, iid in enumerate(trainer_ids):
            out_dir = f"{benchmark['out_dir']}/node_{idx}"
            cmd_id = launcher.send_shell_commands(
                instance_ids=[iid],
                commands=build_trainer_command(
                    repo_dir=code_dir,
                    run_dir=cfg["run_dir"],
                    service=service,
                    mode=benchmark["mode"],
                    model=benchmark["model"],
                    path=benchmark["path"],
                    steps=int(benchmark["steps"]),
                    batch_size=int(benchmark["batch_size"]),
                    num_trainers=int(benchmark["num_trainers_per_node"]),
                    repeat=bool(benchmark.get("repeat", True)),
                    shuffle=bool(benchmark.get("shuffle", False)),
                    trainer_stagger_sec=float(benchmark.get("trainer_stagger_sec", 1.0)),
                    out_dir=out_dir,
                ),
                comment=f"Run benchmark on trainer node {iid}",
                timeout_sec=int(benchmark.get("timeout_sec", 7200)),
            )
            launcher.wait_for_command(
                cmd_id,
                [iid],
                timeout_sec=int(benchmark.get("timeout_sec", 7200)),
            )

        print("\nCluster launch and benchmark run complete.")
        print(json.dumps(launched, indent=2))

    finally:
        if args.terminate_on_finish:
            all_ids = launched["dispatcher"] + launched["workers"] + launched["trainers"]
            launcher.terminate_instances(all_ids)


if __name__ == "__main__":
    main()