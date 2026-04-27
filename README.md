# A/B Test Simulator

Interactive Monte Carlo simulator comparing Frequentist vs Bayesian A/B testing methods.

## Math and Assumptions

For a user-friendly walkthrough of formulas, assumptions, and metric interpretation, see:

- [MATH_ASSUMPTIONS_GUIDE.md](./MATH_ASSUMPTIONS_GUIDE.md)

## Simulation Logic Notes

### Performance optimization for large sample sizes

To prevent browser freezes at very high `samples/day` and long durations, the simulator now samples group means directly instead of generating every individual observation.

- Previous approach (slow): draw `n` samples from `Normal(μ, σ²)`, then compute the sample mean.
- Current approach (fast): draw one value directly from the sampling distribution of the mean:

\[
\bar{X} \sim \text{Normal}\left(\mu, \frac{\sigma^2}{n}\right)
\]

This change is mathematically equivalent under the model assumption already used by the simulator (`Normal(μ, σ²)` observations), but reduces computational complexity from `O(n)` per mean draw to `O(1)`.

Practical impact:
- Same statistical interpretation under the current assumptions
- Significantly better responsiveness for high-traffic / long-duration settings

## Local Development

```bash
npm install
npm run dev
# → http://localhost:5173
```

## Deploy to Vercel (Production + Custom Domain)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "initial commit"
# Create a new repo at github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/ab-simulator.git
git push -u origin main
```

### Step 2 — Deploy via Vercel CLI

```bash
npm install -g vercel
vercel login
vercel
```

Follow the prompts:
- Set up and deploy? → Y
- Which scope? → your account
- Link to existing project? → N
- Project name → ab-simulator
- Directory → ./
- Override settings? → N

Your app is live at `https://ab-simulator-xxx.vercel.app`

### Step 3 — Add Custom Domain

```bash
vercel domains add yourdomain.com
```

Or via the Vercel dashboard:
1. Go to vercel.com → your project → Settings → Domains
2. Add your domain (e.g. `abtester.yourdomain.com`)
3. Copy the DNS records shown (A record or CNAME)
4. Add them in your domain registrar (GoDaddy / Namecheap / Cloudflare etc.)
5. Wait 5–60 min for DNS propagation — Vercel auto-provisions HTTPS

### Step 4 — Auto-deploy on every push

Once connected to GitHub, every `git push origin main` triggers a new production deploy automatically.

```bash
# Future updates:
git add .
git commit -m "update simulation parameters"
git push origin main
# → Live in ~30 seconds
```

## Build for manual deploy (Netlify drag-and-drop)

```bash
npm run build
# Upload the /dist folder at app.netlify.com/drop
```

## Project Structure

```
ab-simulator/
├── index.html          # HTML entry point
├── vite.config.js      # Vite bundler config
├── vercel.json         # Vercel SPA routing + cache headers
├── package.json
├── public/
│   └── favicon.svg
└── src/
    ├── main.jsx        # React root mount
    └── App.jsx         # Full simulator (all logic + charts)
```

## Tech Stack

- React 18
- Vite 5
- Recharts
- Pure browser Monte Carlo (no backend needed)
