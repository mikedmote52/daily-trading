// frontend/src/entry.ts
// Try a few common entry files; load the first one that exists.
async function boot() {
  const candidates = {
    ...import.meta.glob('./main.tsx'),
    ...import.meta.glob('./main.jsx'),
    ...import.meta.glob('./index.tsx'),
    ...import.meta.glob('./index.jsx'),
    ...import.meta.glob('./bootstrap.ts'),
    ...import.meta.glob('./bootstrap.js'),
  };

  const loaders = Object.values(candidates);
  if (loaders.length > 0) {
    // Calling the loader executes the module (side-effect mount).
    await loaders[0]();
    return;
  }

  // Fallback so the build succeeds and you at least see a page.
  const el = document.getElementById('root');
  if (el) el.innerHTML = `<style>body{font-family:ui-sans-serif,system-ui}</style>
  <h1>AlphaStack UI</h1><p>No entry file found. Expected one of:
  <code>src/main.tsx</code>, <code>src/main.jsx</code>, <code>src/index.tsx</code>, <code>src/index.jsx</code>.</p>`;
}
boot();