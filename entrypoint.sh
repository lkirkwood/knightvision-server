#!/usr/bin/env sh

pushd /opt/knightvision-server

caddy reverse-proxy --from :80 --to :8080
