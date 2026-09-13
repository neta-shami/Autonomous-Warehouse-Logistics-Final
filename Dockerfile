# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build an isolated virtualenv with the pinned dependencies.
# ---------------------------------------------------------------------------
FROM python:3.11.9-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: headless simulation. No GPU, no display, no OpenGL required —
# MuJoCo physics stepping is pure compute. This is the default image.
# ---------------------------------------------------------------------------
FROM python:3.11.9-slim-bookworm AS headless

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 sim
WORKDIR /app
COPY --chown=sim:sim . .

USER sim

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mujoco, sys; sys.exit(0)"

CMD ["python", "main.py"]

# ---------------------------------------------------------------------------
# Stage 3: viewer. Adds a virtual X display plus a VNC/noVNC bridge so the
# MuJoCo 3D window can be reached from a browser on any machine.
# ---------------------------------------------------------------------------
FROM headless AS viewer

USER root

# libgl1-mesa-dri supplies the llvmpipe software rasteriser, so the viewer
# renders without any GPU passthrough.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        xvfb \
        x11vnc \
        novnc \
        websockify \
        fluxbox \
        xterm \
        libgl1 \
        libglx-mesa0 \
        libgl1-mesa-dri \
        libglfw3 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:99 \
    SCREEN_GEOMETRY=1600x1000x24 \
    MUJOCO_GL=glfw \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    NOVNC_PORT=6080

COPY --chown=sim:sim docker/start-viewer.sh /usr/local/bin/start-viewer
RUN chmod 0555 /usr/local/bin/start-viewer

USER sim

EXPOSE 6080

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD python -c "import socket,os,sys; s=socket.create_connection(('127.0.0.1', int(os.environ['NOVNC_PORT'])), 5); s.close()"

CMD ["start-viewer"]
