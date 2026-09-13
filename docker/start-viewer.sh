#!/bin/sh
# Boots a virtual X display, publishes it over noVNC, and launches the MuJoCo
# viewer inside a terminal window so the simulator's own command prompt stays
# usable from the browser.
set -eu

DISPLAY_NUM="${DISPLAY#:}"
VNC_PORT=5900
X_SOCKET="/tmp/.X11-unix/X${DISPLAY_NUM}"

pids=""

cleanup() {
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT HUP INT TERM

# -nolisten tcp keeps the X server reachable only through its local socket.
Xvfb "$DISPLAY" -screen 0 "$SCREEN_GEOMETRY" -nolisten tcp >/dev/null 2>&1 &
pids="$pids $!"

waited=0
while [ ! -e "$X_SOCKET" ]; do
    if [ "$waited" -ge 100 ]; then
        echo "Xvfb failed to create $X_SOCKET within 10s" >&2
        exit 1
    fi
    sleep 0.1
    waited=$((waited + 1))
done

fluxbox >/dev/null 2>&1 &
pids="$pids $!"

# -localhost confines the raw VNC port to the container; only the noVNC HTTP
# port is published, and compose binds that to the host loopback.
x11vnc -display "$DISPLAY" -rfbport "$VNC_PORT" -localhost -forever -shared \
    -nopw -quiet >/dev/null 2>&1 &
pids="$pids $!"

# -hold keeps the window open if the simulation exits, so tracebacks stay
# readable on screen. Output is also teed to a file, because a window on a
# virtual display is invisible to `docker compose logs`; inspect it with
# `docker compose exec viewer cat /tmp/sim.log`.
xterm -hold -geometry 100x28+10+620 -title "warehouse sim" \
    -e sh -c 'python visualize.py 2>&1 | tee /tmp/sim.log' >/dev/null 2>&1 &
pids="$pids $!"

echo "Viewer ready. Open http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=1&resize=scale"

exec websockify --web=/usr/share/novnc "$NOVNC_PORT" "127.0.0.1:${VNC_PORT}"
