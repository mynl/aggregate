#!/usr/bin/env bash
# Build the aggregate SPA into src/aggregate/api/static/.
#
# Usage:
#   ./scripts/build-web.sh
#   ./scripts/build-web.sh https://api.mynl.com    # split-origin build

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
web_dir="$script_dir/../web"

api_base="${1:-}"

cd "$web_dir"

if [[ -n "$api_base" ]]; then
    export VITE_API_BASE_URL="$api_base"
    echo "Building with VITE_API_BASE_URL=$api_base"
fi

if [[ ! -d node_modules ]]; then
    echo "Installing npm dependencies..."
    npm install --no-fund --no-audit
fi

echo "Running vite build..."
npm run build
