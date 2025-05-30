import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server

from openpi.training import config as training_config
from openpi.policies import policy_config
from openpi.shared import download

@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    config_name: str = 'pi0'
    checkpoint_bucket: str = '/path/to/your/finetuned/checkpoint/bucket'
    port: int = 23330
    record: bool = False

def main(args: Args) -> None:
    config_name = args.config_name
    checkpoint_bucket = args.checkpoint_bucket
    config = training_config.get_config(config_name)
    checkpoint_dir = download.maybe_download(checkpoint_bucket)

    if not 'pcd' in config_name:
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
    else:
        policy = policy_config.create_pcd_trained_policy(config, checkpoint_dir)
        
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
