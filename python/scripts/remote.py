"""SSH profiles, loopback forwarding, and calibration execution on a Linux host."""

from __future__ import annotations

import json
import os
import select
import shlex
import tempfile
import time
import webbrowser
import zipfile
from getpass import getpass
from pathlib import Path, PurePosixPath
from socketserver import BaseRequestHandler, ThreadingTCPServer
from threading import Thread


def load_profile(path, name):
    profiles = json.loads(Path(path).read_text(encoding="utf-8-sig"))["hosts"]
    if name not in profiles:
        raise ValueError(f"Unknown SSH profile {name!r} in {path}")
    profile = profiles[name]
    if not isinstance(profile, dict):
        raise ValueError(f"SSH profile {name!r} must be an object")
    for key in ("host", "user"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError(f"SSH profile {name!r} requires {key}")
    if profile.get("password") is not None and not isinstance(profile["password"], str):
        raise ValueError(f"SSH profile {name!r}: password must be text or null")
    if (
        type(profile.get("port", 22)) is not int
        or not 1 <= profile.get("port", 22) <= 65535
    ):
        raise ValueError(f"SSH profile {name!r}: invalid port")
    return profile


def connect(host, user, password=None, *, port=22):
    import paramiko

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    try:
        try:
            ssh.connect(host, port=port, username=user, password=password, timeout=10)
        except paramiko.SSHException as error:
            authentication_failed = (
                isinstance(error, paramiko.AuthenticationException)
                or str(error) == "No authentication methods available"
            )
            if password is not None or not authentication_failed:
                raise
            ssh.connect(
                host,
                port=port,
                username=user,
                timeout=10,
                password=getpass(f"{user}@{host} password: "),
            )
    except BaseException:
        ssh.close()
        raise
    return ssh


def run_remote(ssh, *args):
    _, output, errors = ssh.exec_command(shlex.join(args), timeout=20)
    content = output.read().decode(errors="replace").strip()
    if output.channel.recv_exit_status():
        raise RuntimeError(
            errors.read().decode(errors="replace").strip()
            or content
            or f"Remote command failed: {args[0]}"
        )
    return content


def tunnel(ssh, local_port, remote_port):
    import paramiko

    class Forwarder(BaseRequestHandler):
        def handle(self):
            try:
                channel = ssh.get_transport().open_channel(
                    "direct-tcpip",
                    ("127.0.0.1", remote_port),
                    self.client_address,
                    timeout=5,
                )
                with channel:
                    while True:
                        readable, _, _ = select.select([self.request, channel], [], [])
                        for source in readable:
                            target = channel if source is self.request else self.request
                            data = source.recv(65536)
                            if not data:
                                return
                            target.sendall(data)
            except (OSError, EOFError, paramiko.SSHException):
                return

    server = ThreadingTCPServer(("127.0.0.1", local_port), Forwarder)
    server.daemon_threads = True
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def add_target_arguments(parser, root):
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--local", action="store_true", help="Run locally (default)")
    target.add_argument("--ssh", metavar="PROFILE", help="Run on a Linux SSH host")
    parser.add_argument(
        "--ssh-config",
        type=Path,
        default=root / "configs/ssh.local.json",
        help="SSH profile JSON (defaults to configs/ssh.local.json)",
    )


def remote_path(value, root):
    path = Path(value)
    if path.is_absolute() and path.is_relative_to(root):
        path = path.relative_to(root)
    if path.drive:
        raise ValueError(f"Path {value} must refer to the remote Linux host")
    return path.as_posix()


def run_workflow(args, root, *, marker=False):
    command_args = [args.command, "--config", remote_path(args.config, root)]
    for flag, value in (
        ("--session", args.session),
        ("--calibration-dir", getattr(args, "calibration_dir", None)),
        ("--camera-intrinsics", getattr(args, "camera_intrinsics", None)),
    ):
        if value is not None:
            command_args.extend((flag, remote_path(value, root)))
    script = "calibrate_marker.py" if marker else "calibrate.py"
    if camera := getattr(args, "camera", None):
        command_args.extend(("--camera", camera))
    return run_calibration(args, root, command_args, script=script)


def run_calibration(args, root, command_args, *, script=None, module=None):
    profile = load_profile(args.ssh_config, args.ssh)
    project = profile.get("project_dir", "")
    if not isinstance(project, str) or not PurePosixPath(project).is_absolute():
        raise ValueError("SSH profile requires an absolute Linux project_dir")
    python = profile.get("python", "python3")
    import paramiko

    try:
        with connect(
            profile["host"],
            profile["user"],
            profile.get("password"),
            port=profile.get("port", 22),
        ) as ssh:
            try:
                run_remote(ssh, "test", "-d", project)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Remote project directory is unavailable: {project}"
                ) from error
            directory = run_remote(ssh, "mktemp", "-d", "/tmp/calibration-ssh.XXXXXXXX")
            remote_directory = PurePosixPath(directory)
            if remote_directory.parent != PurePosixPath(
                "/tmp"
            ) or not remote_directory.name.startswith("calibration-ssh."):
                raise RuntimeError("SSH host did not return a valid runtime directory")
            server = None
            try:
                with tempfile.TemporaryFile() as archive:
                    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                        for path in (root / "python").rglob("*"):
                            if path.is_file() and path.suffix in {".py", ".html", ".js"}:
                                bundle.write(path, path.relative_to(root).as_posix())
                    archive.seek(0)
                    with ssh.open_sftp() as sftp:
                        sftp.putfo(archive, f"{directory}/runtime.zip")
                run_remote(
                    ssh,
                    python,
                    "-m",
                    "zipfile",
                    "-e",
                    f"{directory}/runtime.zip",
                    directory,
                )
                if module:
                    entry = [
                        "-c",
                        "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
                        "runpy.run_module(sys.argv.pop(1), run_name='__main__')",
                        directory,
                        module,
                    ]
                else:
                    entry = [f"{directory}/python/scripts/{script}"]
                invocation = shlex.join(
                    [
                        "env",
                        f"CALIBRATION_PROJECT_ROOT={project}",
                        "CALIBRATION_CAPTURE_WEB=1",
                        f"CALIBRATION_EMBEDDED={os.environ.get('CALIBRATION_EMBEDDED', '0')}",
                        "PYTHONIOENCODING=utf-8",
                        python,
                        "-u",
                        *entry,
                        *command_args,
                    ]
                )
                print(
                    f"SSH calibration: {args.ssh} ({profile['host']}), project {project}",
                    flush=True,
                )
                _, output, _ = ssh.exec_command(
                    f"cd {shlex.quote(project)} && exec {invocation}", get_pty=True
                )
                try:
                    for line in output:
                        line = line.rstrip("\r\n")
                        if line.startswith("CALIBRATION_CAPTURE_PORT="):
                            port = int(line.partition("=")[2])
                            server = tunnel(ssh, 0, port)
                            url = f"http://127.0.0.1:{server.server_address[1]}/"
                            print(f"Remote capture: {url}", flush=True)
                            if not os.environ.get("CALIBRATION_EMBEDDED"):
                                webbrowser.open(url)
                        else:
                            print(line, flush=True)
                    return output.channel.recv_exit_status()
                except KeyboardInterrupt:
                    try:
                        output.channel.sendall(b"\x03")
                        deadline = time.monotonic() + 5
                        while not output.channel.exit_status_ready() and time.monotonic() < deadline:
                            time.sleep(0.1)
                    except (OSError, paramiko.SSHException):
                        pass
                    output.channel.close()
                    return 130
            finally:
                if server:
                    server.shutdown()
                    server.server_close()
                if ssh.get_transport().is_active():
                    run_remote(ssh, "rm", "-rf", "--", directory)
    except paramiko.SSHException as error:
        raise RuntimeError(f"SSH {args.ssh}: {error}") from error
