#!/bin/bash
# Render build script for AlphaStack Frontend

echo "🔧 Starting AlphaStack Frontend build..."
echo "📁 Current directory: $(pwd)"
echo "📋 Directory contents:"
ls -la

echo "📦 Installing dependencies..."
npm install --legacy-peer-deps

echo "🏗️ Building React application..."
npm run build

echo "✅ Build complete!"
echo "📂 Build directory contents:"
ls -la build/

echo "📄 Built index.html:"
head -5 build/index.html