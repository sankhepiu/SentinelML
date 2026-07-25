# syntax=docker/dockerfile:1
#
# Build context is this directory (frontend/) -- a plain npm project, no
# workspace dependency on the rest of the repo.

ARG NODE_VERSION=22
ARG NGINX_VERSION=1.27-alpine

# ---- builder ----------------------------------------------------------
FROM node:${NODE_VERSION}-slim AS builder
WORKDIR /app

# Install dependencies first, before copying source, so this (slow) layer
# is cached across source-only changes.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
# VITE_API_BASE_URL defaults to /api/v1 (relative), which nginx.conf below
# proxies to the backend -- no build-time API URL needed for the default
# same-origin deployment. Pass --build-arg to bake in a different one
# (e.g. a separately hosted backend) if needed.
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# ---- runtime ------------------------------------------------------------
FROM nginx:${NGINX_VERSION} AS runtime
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
