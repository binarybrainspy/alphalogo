# AlphaLogo Website — Deployment Guide

## File Structure Needed

```
your-project/
├── index.html              ← the website (this file)
├── logograms/              ← your 49 PNG files
│   ├── 000.png
│   ├── 001.png
│   └── ... (through 048.png)
└── logograms_wire/         ← optional: skeletal reconstructions
    ├── 000.png
    └── ...
```

## Deploy to Vercel (recommended)

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. In your project folder:
   ```bash
   vercel
   ```
   Follow the prompts. Select "No framework" when asked.

3. Your site is live at `https://your-project.vercel.app`

## Deploy to Netlify

### Option A — Drag and drop (easiest)
1. Go to [netlify.com](https://netlify.com) → Log in
2. Drag your entire project folder onto the Netlify dashboard
3. Done — live in seconds

### Option B — Git deploy
1. Push your project to GitHub
2. Connect the repo to Netlify
3. Build command: (leave empty)
4. Publish directory: `.` (root)

## Notes

- The `logograms/` folder must contain your 49 PNG files named
  `000.png` through `048.png` exactly as they appear in your dataset
- The skeletal wireframe images (`logograms_wire/`) are optional —
  if absent, the wireframe tab in the modal will fall back to the original
- No backend or build step required — pure HTML/CSS/JS
- All fonts load from Google Fonts (requires internet)

## Custom Domain

In Vercel: Project Settings → Domains → Add your domain
In Netlify: Site Settings → Domain Management → Add custom domain