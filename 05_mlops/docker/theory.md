# Docker

## What problem does Docker solve?

When you build a machine learning application on your laptop, it works perfectly. But when you deploy it to a server — or share it with a colleague — it breaks. The error message usually looks something like this:

```
ModuleNotFoundError: No module named 'sklearn'
```

The root cause is always the same: **the environment is different**. Your laptop has Python 3.11, the server has Python 3.9. You installed `scikit-learn` in a conda environment, but the server doesn't know that. You have a specific version of `joblib` that is incompatible with theirs.

This is the classic _"it works on my machine"_ problem.

Docker solves this by packaging your application **together with its entire environment** — the Python version, all dependencies, the OS libraries, everything — into a single portable unit. That unit runs identically on any machine that has Docker installed, whether it's your laptop, a colleague's computer, or a cloud server.

Think of it like this: instead of shipping a recipe and hoping the other kitchen has the same ingredients and tools, you ship the entire prepared meal in a sealed container.

---

## Core concepts

### Image

A Docker **image** is a read-only blueprint that describes everything needed to run your application:

- The base operating system (e.g. Ubuntu, or a slim Python image)
- System-level dependencies
- Your Python version and all installed packages
- Your application code
- The command to run when the container starts

An image is built **once** from a set of instructions (the Dockerfile). It is not running — it is just a static snapshot, like a class definition in Python before you instantiate it.

Images are **layered**. Each instruction in the Dockerfile adds a new layer on top of the previous one. Docker caches these layers, so rebuilding after a small change is fast: only the changed layers and those after them are rebuilt.

You can share images by pushing them to a **registry** like [Docker Hub](https://hub.docker.com/) or GitHub Container Registry, just as you push code to GitHub.

### Container

A **container** is a running instance of an image — the image brought to life.

If an image is the class, the container is the object. You can create multiple containers from the same image, each running in isolation from the others.

Containers are:

- **Isolated**: they have their own filesystem, network, and process space, separate from the host machine and from each other
- **Lightweight**: they share the host OS kernel, unlike virtual machines which emulate an entire OS
- **Ephemeral by default**: when a container stops, any data written inside it is lost (unless you use volumes)

This isolation is what makes Docker so reliable. Your FastAPI app running inside a container cannot be affected by anything happening outside of it.

### Dockerfile

A **Dockerfile** is a plain text file containing a sequence of instructions that tell Docker how to build an image.

It is the source of truth for your environment. You check it into version control alongside your code, and anyone can reproduce your exact environment by running `docker build`.

---

## Dockerfile instructions

Here is a complete Dockerfile for a FastAPI ML service, with each instruction explained:

```dockerfile
# 1. FROM — choose your base image
FROM python:3.12-slim
```

Every Dockerfile starts with `FROM`. It specifies the **base image** to build on top of. `python:3.12-slim` is an official Python image with Python 3.12 pre-installed, based on a minimal version of Debian Linux (`slim` means unnecessary tools have been removed to keep the image small).

You never start from scratch — you always build on a base.

---

```dockerfile
# 2. WORKDIR — set the working directory inside the container
WORKDIR /app
```

`WORKDIR` creates a directory inside the container and sets it as the current working directory for all subsequent instructions. Think of it as `cd /app` that also creates the folder if it doesn't exist.

All relative paths in the rest of the Dockerfile will be relative to `/app`.

---

```dockerfile
# 3. COPY — copy files from your machine into the image
COPY requirements.txt .
```

`COPY <source> <destination>` copies files from your local filesystem (the **build context**) into the image. Here, `requirements.txt` is copied to the current working directory (`.`, which is `/app`).

We copy `requirements.txt` _before_ the rest of the code on purpose — this is a key layer-caching optimisation explained below.

---

```dockerfile
# 4. RUN — execute a shell command during the build
RUN pip install --no-cache-dir -r requirements.txt
```

`RUN` executes a command inside the image at build time. Here it installs all Python dependencies.

`--no-cache-dir` tells pip not to store its download cache inside the image, which keeps the image smaller.

This layer is expensive (it downloads and installs packages). Because it comes _after_ the `COPY requirements.txt` step and _before_ the `COPY . .` step, Docker can cache it: as long as `requirements.txt` hasn't changed, Docker will reuse this layer on the next build without reinstalling everything.

---

```dockerfile
# 5. COPY the rest of the application code
COPY . .
```

Now we copy all remaining files (your `main.py`, model `.pkl` files, etc.) into `/app`. This is done _after_ the dependency installation step so that changing your application code doesn't invalidate the package installation cache.

---

```dockerfile
# 6. EXPOSE — document which port the container listens on
EXPOSE 8000
```

`EXPOSE` is **documentation only** — it tells Docker (and humans reading the file) that the application inside the container listens on port 8000. It does not actually publish the port to the host machine; that is done at runtime with `-p`.

---

```dockerfile
# 7. CMD — the default command to run when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`CMD` defines the command that runs when a container is started from this image. The list form (called _exec form_) is preferred over the string form because it doesn't invoke a shell, which means signals like `Ctrl+C` are handled correctly.

`--host 0.0.0.0` is essential: by default, Uvicorn only listens on `localhost` inside the container, which is unreachable from outside. `0.0.0.0` means "accept connections from any address".

---

### The complete Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Why this order matters — layer caching

Docker builds images layer by layer and caches each one. If a layer hasn't changed since the last build, Docker reuses the cached version and skips rebuilding it.

The rule is: **put things that change rarely near the top, and things that change often near the bottom.**

- `requirements.txt` changes rarely → install dependencies early
- Your application code changes constantly → copy it last

If you did it the other way around (`COPY . .` first, then `RUN pip install`), every time you changed a single line in `main.py`, Docker would invalidate the cache and reinstall all packages from scratch. With the correct order, only the `COPY . .` layer and `CMD` need to be rerun.

---

## Key commands

### Building an image

```bash
docker build -t my-ml-app .
```

- `build` — build an image from a Dockerfile
- `-t my-ml-app` — tag (name) the image `my-ml-app`
- `.` — the build context (the current directory, where Docker looks for the Dockerfile and the files to `COPY`)

### Running a container

```bash
docker run -p 8000:8000 my-ml-app
```

- `run` — create and start a container from an image
- `-p 8000:8000` — map port 8000 on your machine to port 8000 inside the container (`-p host_port:container_port`)
- `my-ml-app` — the image to use

### Running in detached mode (background)

```bash
docker run -d -p 8000:8000 --name ml-service my-ml-app
```

- `-d` — detached mode: the container runs in the background
- `--name ml-service` — give the container a readable name

### Viewing running containers

```bash
docker ps
```

### Viewing logs from a running container

```bash
docker logs ml-service
```

### Stopping a container

```bash
docker stop ml-service
```

### Removing a container

```bash
docker rm ml-service
```

### Listing images

```bash
docker images
```

### Removing an image

```bash
docker rmi my-ml-app
```

### Opening a shell inside a running container (very useful for debugging)

```bash
docker exec -it ml-service /bin/bash
```

- `exec` — run a command inside a running container
- `-it` — interactive terminal
- `/bin/bash` — open a bash shell

---

## Mental model summary

| Concept               | Analogy                                       |
| --------------------- | --------------------------------------------- |
| Dockerfile            | Recipe                                        |
| Image                 | Meal prepared and vacuum-sealed               |
| Container             | Meal served and being eaten                   |
| Registry (Docker Hub) | Supermarket where you share or download meals |

```
Dockerfile  ──[docker build]──►  Image  ──[docker run]──►  Container
                                         ◄─[docker stop]──
```
