# mavleo96.github.io

Personal academic portfolio for **Vijayabharathi Murugan** — ML engineer and graduate researcher at Stony Brook University.

Live at: **https://mavleo96.github.io**

Built with [Jekyll](https://jekyllrb.com/) and the [al-folio](https://github.com/alshedivat/al-folio) theme, deployed to GitHub Pages.

---

## System requirements

- **Ruby 3.3.5** — macOS ships with 2.6 which is too old. Install via [rbenv](https://github.com/rbenv/rbenv) or [RVM](https://rvm.io/):
  ```bash
  rbenv install 3.3.5 && rbenv local 3.3.5
  ```
- **Node.js** (any current LTS)
- **Python 3** (for pre-commit)
- **ImageMagick** — used by `jekyll-imagemagick` to generate responsive images:
  ```bash
  brew install imagemagick
  ```

## Setup

```bash
# Install Ruby dependencies
bundle install

# Install Node dependencies (prettier + Liquid plugin)
npm install

# Install and register the pre-commit hooks
pip install pre-commit
pre-commit install

# (Optional) install nbconvert for Jupyter notebook post support
pip install nbconvert
```

## Running locally

```bash
bundle exec jekyll serve
```

Site is served at `http://localhost:4000`. GoatCounter analytics are skipped in dev — set `JEKYLL_ENV=production` to test the production build locally.

---

## Content files

| What            | Where                      |
| --------------- | -------------------------- |
| Bio / homepage  | `_pages/about.md`          |
| Publications    | `_bibliography/papers.bib` |
| Projects        | `_projects/*.md`           |
| News items      | `_news/*.md`               |
| Blog posts      | `_posts/*.md`              |
| CV data         | `assets/json/resume.json`  |
| CV page         | `_pages/cv.md`             |
| Social links    | `_data/socials.yml`        |
| Co-author links | `_data/coauthors.yml`      |
| Site config     | `_config.yml`              |

---

## CI / CD

Three GitHub Actions workflows run on push to `main`:

- **deploy** — builds the Jekyll site and pushes to the `gh-pages` branch
- **prettier** — checks formatting of Liquid, JS, CSS, YAML, and Markdown
- **broken-links** — scans source files for dead URLs

Prettier also runs as a pre-commit hook via [`pre-commit`](https://pre-commit.com/).

**GitHub Pages setup**: in repo Settings → Pages, set source to the `gh-pages` branch (root). The `deploy` workflow creates and pushes to it automatically.

---

## Analytics

Visitor tracking via [GoatCounter](https://www.goatcounter.com/) — privacy-friendly, no cookies.
Dashboard at **https://mavleo96.goatcounter.com**.
Only active in production (`JEKYLL_ENV=production`); not loaded during local dev.

---

## License

MIT
