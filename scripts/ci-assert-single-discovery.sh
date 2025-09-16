#!/usr/bin/env bash
# CI Guard Script - Enforce Single Discovery System
# Prevents creation of duplicate discovery, scanner, or finder systems

set -euo pipefail

echo "🔍 Checking for duplicate discovery systems..."

# Find all discovery-like files that actually exist
ALL_DISCOVERY=$(find . -name "*.py" -type f | grep -E '(discovery|finder|scanner)' \
  | grep -v -E '/venv/' \
  | grep -v -E '/__pycache__/' \
  | sed 's|^\./||' || true)

# Filter out allowed files
BAD=""
for file in $ALL_DISCOVERY; do
  case "$file" in
    "agents/discovery/universal_discovery.py"|"agents/discovery/discovery_api.py")
      # These are allowed
      ;;
    "agents/discovery/config.py"|"agents/discovery/models.py"|"agents/discovery/startup.sh")
      # These are allowed configuration files
      ;;
    "discovery_pipeline_tracer.py"|"run_live_discovery.py"|"optimized_discovery_test.py")
      # These are allowed test/utility files
      ;;
    "agents/discovery/full_live_test.py"|"agents/discovery/live_filter_test.py")
      # These are allowed test files in discovery directory
      ;;
    *)
      # Everything else is forbidden
      if [ -z "$BAD" ]; then
        BAD="$file"
      else
        BAD="$BAD
$file"
      fi
      ;;
  esac
done

if [ -n "$BAD" ]; then
  echo "❌ ERROR: Multiple discovery-like files detected!"
  echo "Only agents/discovery/universal_discovery.py is allowed as the discovery engine."
  echo ""
  echo "Forbidden files found:"
  echo "$BAD"
  echo ""
  echo "🚨 SYSTEM RULE VIOLATION:"
  echo "- ONE DISCOVERY SYSTEM ONLY: agents/discovery/universal_discovery.py"
  echo "- NO duplicates, backups, or enhanced versions allowed"
  echo "- Edit the existing file directly instead of creating new ones"
  exit 1
fi

echo "✅ Single discovery system verified: agents/discovery/universal_discovery.py"
echo "✅ No duplicate discovery systems found"