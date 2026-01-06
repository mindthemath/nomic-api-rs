# GitHub Actions Workflows

## Docker Build and Push

The `docker.yml` workflow automatically builds and pushes Docker images to DockerHub when:
- A tag is pushed (e.g., `v1.0.0`)
- Manually triggered via GitHub Actions UI

**Required Secrets:**
- `DOCKERHUB_USERNAME` - Your DockerHub username
- `DOCKERHUB_TOKEN` - DockerHub access token (create at https://hub.docker.com/settings/security)

**Images pushed:**
- `mindthemath/nomic-text-v1.5-rs:<tag>-cpu` - CPU-only image
- `mindthemath/nomic-text-v1.5-rs:<tag>-gpu` - GPU/CUDA image
- `mindthemath/nomic-text-v1.5-rs:latest-cpu` - Latest CPU image
- `mindthemath/nomic-text-v1.5-rs:latest-gpu` - Latest GPU image

