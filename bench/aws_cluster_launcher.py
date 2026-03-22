#!/usr/bin/env python3
import argparse
import json
import sys
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
        tags = [
            {"Key": "Name", "Value": f"{cluster_name}-{role}"},
            {"Key": "Cluster", "Value": cluster_name},
            {"Key": "Role", "Value": role},
            {"Key": "ManagedBy", "Value": "aws_cluster_launcher"},
        ]
        if extra_tags:
            for k, v in extra_tags.items():
                tags.append({"Key": k, "Value": v})

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

    def wait_for_instances_running(self, instance_ids: List[str], timeout_sec: int = 900) -> None:
        waiter = self.ec2.get_waiter("instance_running")
        waiter.wait(
            InstanceIds=instance_ids,
            WaiterConfig={"Delay": 10, "MaxAttempts": max(1, timeout_sec // 10)},
        )
        print(f"Instances are running: {instance_ids}")

    def describe_instances(self, instance_ids: List[str]) -> List[Dict]:
        resp = self.ec2.describe_instances(InstanceIds=instance_ids)
        out = []
        for r in resp["Reservations"]:
            for inst in r["Instances"]:
                out.append(inst)
        return out

    def get_private_ip_map(self, instance_ids: List[str]) -> Dict[str, str]:
        details = self.describe_instances(instance_ids)
        return {
            inst["InstanceId"]: inst.get("PrivateIpAddress", "")
            for inst in details
        }

    def wait_for_ssm_online(self, instance_ids: List[str], timeout_sec: int = 900) -> Dict[str, str]:
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
                inv = self.ssm.get_command_invocation(CommandId=command_id, InstanceId=iid)
                status = inv["Status"]
                if status in {"Success"}:
                    done.add(iid)
                elif status in {"Pending", "InProgress", "Delayed"}:
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

    def terminate_instances(self, instance_ids: List[str]) -> None:
        if not instance_ids:
            return
        self.ec2.terminate_instances(InstanceIds=instance_ids)
        print(f"Terminate requested for: {instance_ids}")


def build_dispatcher_command(code_dir: str, dispatcher_port: int, work_dir: str) -> List[str]:
    return [
        f"cd {code_dir}",
        "mkdir -p logs",
        (
            f"nohup python dispatcher.py "
            f"--port {dispatcher_port} "
            f"--work-dir {work_dir} "
            f"> logs/dispatcher.log 2>&1 &"
        ),
        "sleep 2",
        "ps -ef | grep dispatcher.py | grep -v grep",
    ]


def build_worker_commands(
    code_dir: str,
    dispatcher_address: str,
    worker_ports: List[int],
    worker_host: str,
) -> List[str]:
    cmds = [f"cd {code_dir}", "mkdir -p logs"]
    for port in worker_ports:
        cmds.append(
            f"nohup python worker.py "
            f"--dispatcher-address {dispatcher_address} "
            f"--port {port} "
            f"--worker-address {worker_host}:{port} "
            f"> logs/worker_{port}.log 2>&1 &"
        )
    cmds.extend(
        [
            "sleep 3",
            "ps -ef | grep worker.py | grep -v grep",
        ]
    )
    return cmds


def build_trainer_command(
    code_dir: str,
    service: str,
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
    repeat_flag = "--repeat" if repeat else ""
    shuffle_flag = "--shuffle" if shuffle else ""

    cmd = (
        f"cd {code_dir} && mkdir -p {out_dir} && "
        f"python run_benchmark.py "
        f"--mode {mode} "
        f"--service {service} "
        f"--model {model} "
        f"--path {path} "
        f"--steps {steps} "
        f"--batch-size {batch_size} "
        f"--num-trainers {num_trainers} "
        f"--trainer-stagger-sec {trainer_stagger_sec} "
        f"--out-dir {out_dir} "
        f"{repeat_flag} {shuffle_flag}"
    )
    return [
        cmd,
        f"ls -R {out_dir} || true",
    ]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to cluster config JSON")
    parser.add_argument("--terminate-on-finish", action="store_true")
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
        # 1) Launch dispatcher
        dispatcher_ids = launcher.launch_instances(
            spec=dispatcher_spec,
            cluster_name=cluster_name,
            role="dispatcher",
        )
        launched["dispatcher"] = dispatcher_ids
        launcher.wait_for_instances_running(dispatcher_ids)
        launcher.wait_for_ssm_online(dispatcher_ids)
        dispatcher_ip = launcher.get_private_ip_map(dispatcher_ids)[dispatcher_ids[0]]

        # 2) Start dispatcher process
        cmd_id = launcher.send_shell_commands(
            instance_ids=dispatcher_ids,
            commands=build_dispatcher_command(
                code_dir=code_dir,
                dispatcher_port=dispatcher_port,
                work_dir=dispatcher_work_dir,
            ),
            comment="Start tf.data dispatcher",
        )
        launcher.wait_for_command(cmd_id, dispatcher_ids)

        dispatcher_address = f"{dispatcher_ip}:{dispatcher_port}"
        service = f"grpc://{dispatcher_address}"
        print(f"Dispatcher service: {service}")

        # 3) Launch worker nodes
        worker_ids = launcher.launch_instances(
            spec=worker_spec,
            cluster_name=cluster_name,
            role="worker",
        )
        launched["workers"] = worker_ids
        launcher.wait_for_instances_running(worker_ids)
        launcher.wait_for_ssm_online(worker_ids)
        worker_ip_map = launcher.get_private_ip_map(worker_ids)

        # 4) Start worker processes
        for iid in worker_ids:
            worker_host = worker_ip_map[iid]
            cmd_id = launcher.send_shell_commands(
                instance_ids=[iid],
                commands=build_worker_commands(
                    code_dir=code_dir,
                    dispatcher_address=dispatcher_address,
                    worker_ports=worker_ports_per_node,
                    worker_host=worker_host,
                ),
                comment=f"Start tf.data workers on {iid}",
            )
            launcher.wait_for_command(cmd_id, [iid])

        # 5) Launch trainer nodes
        trainer_ids = launcher.launch_instances(
            spec=trainer_spec,
            cluster_name=cluster_name,
            role="trainer",
        )
        launched["trainers"] = trainer_ids
        launcher.wait_for_instances_running(trainer_ids)
        launcher.wait_for_ssm_online(trainer_ids)

        # 6) Start benchmark on trainer nodes
        # Here each trainer node runs one run_benchmark.py process.
        # If you launch multiple trainer nodes, each node will run its own local set of trainer processes.
        for idx, iid in enumerate(trainer_ids):
            out_dir = f"{benchmark['out_dir']}/node_{idx}"
            cmd_id = launcher.send_shell_commands(
                instance_ids=[iid],
                commands=build_trainer_command(
                    code_dir=code_dir,
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