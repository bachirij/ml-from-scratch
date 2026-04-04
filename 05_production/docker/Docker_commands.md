# Docker — Terminal Commands Reference

> Part of the `ml-from-scratch` project  
> Path: `05_production/docker/docker_commands.md`

---

## Prerequisites

Docker Desktop must be installed and running before any of these commands work.

Check that Docker is available:

```bash
docker --version
```

---

## 0. Navigate to your project folder

You must be in the folder that contains your `Dockerfile` before running any
`docker build` or `docker run` commands.

```bash
cd path/to/your/project
```

---

## 1. Build the image

```bash
docker build -t your-image-name .
```

- `docker build` → reads the Dockerfile and builds an image layer by layer
- `-t your-image-name` → gives the image a name (tag) so you can refer to it later
- `.` → tells Docker to look for the Dockerfile in the current directory

**Example:**

```bash
docker build -t diabetes-api .
```

You should see Docker pulling the base image, installing dependencies, and copying
your files. The first build takes longer — subsequent builds reuse cached layers.

---

## 2. Run a container

```bash
docker run -p 8000:8000 your-image-name
```

- `docker run` → creates and starts a container from the image
- `-p 8000:8000` → maps port 8000 on your machine to port 8000 inside the container
  - Format: `-p host_port:container_port`
  - Without this, the container is isolated and unreachable from your browser
- `your-image-name` → the name you gave the image with `-t`

**Example:**

```bash
docker run -p 8000:8000 diabetes-api
```

Once running, the API is available at:

```
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

---

## 3. Run a container in the background (detached mode)

```bash
docker run -d -p 8000:8000 your-image-name
```

- `-d` → detached mode — the container runs in the background, the terminal is free

---

## 4. List running containers

```bash
docker ps
```

Shows all currently running containers with their ID, image name, and port mappings.

---

## 5. Stop a container

If running in the foreground (attached):

```bash
CTRL+C
```

If running in the background (detached), first get the container ID with `docker ps`,
then:

```bash
docker stop <container_id>
```

Or stop all containers running from a specific image:

```bash
docker stop $(docker ps -q --filter ancestor=your-image-name)
```

---

## 6. List all images

```bash
docker images
```

Shows all images on your machine with their name, tag, and size.

---

## 7. Remove an image

```bash
docker rmi your-image-name
```

Useful to free up disk space or force a clean rebuild.

---

## 8. Rebuild after code changes

When you modify your code, you need to rebuild the image:

```bash
docker build -t your-image-name .
docker run -p 8000:8000 your-image-name
```

Docker caches unchanged layers (e.g. pip install) so the rebuild is fast if only
your code files changed.

---

## Full Workflow — From Scratch to Running API

```bash
# 1. Navigate to project folder
cd path/to/your/project

# 2. Build the image
docker build -t your-image-name .

# 3. Run the container
docker run -p 8000:8000 your-image-name

# 4. Test the API
# Open http://127.0.0.1:8000/health in your browser

# 5. Stop the container
CTRL+C
```

---

## References

- [Docker — Official Documentation](https://docs.docker.com/)
- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)
- [docker run reference](https://docs.docker.com/engine/reference/run/)
