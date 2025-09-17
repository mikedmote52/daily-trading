// Dynamically import whichever entry your app uses.
// If you already have src/main.tsx OR src/main.jsx mounting the app, this will pick it up.
(async () => {
  try {
    await import('./main.tsx');
  } catch {
    try {
      await import('./main.jsx');
    } catch (e) {
      console.error('No main.tsx or main.jsx found. Expected one of these:', e);
    }
  }
})();