# Formula Career Predictor

[![codecov](https://codecov.io/gh/jakmate/formula-ascent/branch/main/graph/badge.svg)](https://codecov.io/gh/jakmate/formula-ascent)
[![CI/CD](https://github.com/jakmate/formula-ascent/workflows/CI%2FCD/badge.svg)](https://github.com/jakmate/formula-ascent/actions/workflows/ci.yml)
![Lighthouse](https://raw.githubusercontent.com/jakmate/formula-ascent/main/badges/lighthouse-badge.svg)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=jakmate_formula-ascent&metric=alert_status)](https://sonarcloud.io/summary/overall?id=jakmate_formula-ascent&branch=main)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=jakmate_formula-ascent&metric=security_rating)](https://sonarcloud.io/summary/overall?id=jakmate_formula-ascent&branch=main)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=jakmate_formula-ascent&metric=reliability_rating)](https://sonarcloud.io/summary/overall?id=jakmate_formula-ascent&branch=main)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=jakmate_formula-ascent&metric=sqale_rating)](https://sonarcloud.io/summary/overall?id=jakmate_formula-ascent&branch=main)

A machine learning application that predicts drivers' likelihood of advancing to parent series using historical performance data.

## To Do

### Academies scraper
- Integrate into the workflow

## Deployment

The application is containerized and deployed on:

- **Backend**: FastAPI on Render
- **Frontend**: Vite on Render
- **CI/CD**: GitHub Actions for testing
- **cron-job.org** - Keep instance alive due to Render's spindown with the free tier

## Tech Stack

- **FastAPI** - REST API framework
- **React** - UI framework

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 20+

### Backend Setup

```bash
cd backend
# no gpu
uv sync --extra cpu
# with gpu
uv sync --extra cu128
fastapi run
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Development

### Running Tests

```bash
# Backend
cd backend
pytest --cov=. --cov-report=html --cov-report=term-missing -n auto

# Frontend
cd frontend
npm run test:coverage
```

### Code Quality

```bash
# Backend linting
flake8 .
black --line-length 90 .

# Frontend linting
npx eslint . --ext .js,.jsx,.ts,.tsx
npx prettier --check "src/**/*.{js,jsx,ts,tsx,json,css,md}"
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Acknowledgments

- Official Formula 1, Formula 2, and Formula 3 championships for schedule data
- Wikipedia contributors for driver and team information
