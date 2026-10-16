"""UDP listener that starts/stops the dewu-shoe-insole run.

    python scripts/dewu_udp/udp_listener.py [--port 4560] [--allow-ip 192.168.1.10]

Commands (exact match, UTF-8, surrounding whitespace ignored):
    START  launch the run (ignored if already running)
    STOP   send SIGINT (= one Ctrl+C: graceful stop, arms return home)

The payload is only ever compared against fixed strings, never passed to a shell.
"""

import argparse
import logging
import pathlib
import signal
import socket
import subprocess
import time

SCRIPT = pathlib.Path(__file__).with_name("run_dewu_shoe_insole.sh")
START_CMD = "START"
STOP_CMD = "STOP"
START_COOLDOWN_S = 3.0  # drop duplicate/bounced START packets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dewu_udp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=4560)
    ap.add_argument("--allow-ip", action="append", default=[], help="accepted source IP (repeatable); default: any")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    log.info("listening on udp %s:%d (allow=%s)", args.host, args.port, args.allow_ip or "any")

    proc: subprocess.Popen | None = None
    last_start = 0.0
    sigint_sent = False

    sock.settimeout(1.0)  # wake periodically so a finished run is reaped and logged promptly

    while True:
        try:
            data, (ip, _) = sock.recvfrom(1024)
        except TimeoutError:
            data = None
        running = proc is not None and proc.poll() is None
        if proc is not None and not running:
            log.info("run exited with code %s", proc.returncode)
            proc, sigint_sent = None, False
        if data is None:
            continue
        if args.allow_ip and ip not in args.allow_ip:
            log.warning("ignored packet from %s", ip)
            continue
        cmd = data.decode("utf-8", errors="replace").strip()

        if cmd == START_CMD:
            if running or time.monotonic() - last_start < START_COOLDOWN_S:
                log.info("START ignored (already running)")
                continue
            # New session: Ctrl+C on the listener must not hit the arm process.
            proc = subprocess.Popen([str(SCRIPT)], stdin=subprocess.DEVNULL, start_new_session=True)
            last_start, sigint_sent = time.monotonic(), False
            log.info("started pid=%d", proc.pid)
        elif cmd == STOP_CMD:
            if not running:
                log.info("STOP ignored (not running)")
            elif sigint_sent:
                # A second SIGINT makes main.py os._exit(), skipping the home move.
                log.info("STOP ignored (graceful shutdown already in progress)")
            else:
                proc.send_signal(signal.SIGINT)
                sigint_sent = True
                log.info("sent SIGINT to pid=%d", proc.pid)
        else:
            log.warning("unknown command from %s: %r", ip, cmd[:50])


if __name__ == "__main__":
    main()
