# Contributing to NFL Prediction Model

Thank you for your interest in contributing to the NFL Prediction Model! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/nfl-prediction-model.git
   cd nfl-prediction-model
   ```
3. **Set up the development environment**:
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

## Development Workflow

1. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes** thoroughly:
   ```bash
   # Run the pipeline
   python run_pipeline.py --full
   
   # Test specific components
   python src/fetch_data.py
   python src/build_features.py
   python src/train.py
   ```

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Maximum line length: 100 characters

### Code Formatting

We use `black` for code formatting:

```bash
# Install black
pip install black

# Format your code
black src/
```

### Linting

We use `flake8` for linting:

```bash
# Install flake8
pip install flake8

# Check your code
flake8 src/
```

### Type Hints

Use type hints where appropriate:

```python
def calculate_stats(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Calculate rolling statistics"""
    pass
```

## Documentation

- Add docstrings to all public functions and classes
- Update README.md if adding new features
- Update QUICKSTART.md if changing user-facing behavior
- Add comments for complex logic

### Docstring Format

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of function
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
    """
    pass
```

## Testing

### Manual Testing

Before submitting a PR, test:

1. **Data Pipeline**: Run `python run_pipeline.py --fetch --features`
2. **Model Training**: Run `python run_pipeline.py --train`
3. **Predictions**: Run `python run_pipeline.py --predict`
4. **Dashboard**: Launch with `streamlit run src/dashboard.py`
5. **API**: Start with `python src/api.py` and test endpoints

### Automated Testing (Future)

We plan to add:
- Unit tests with pytest
- Integration tests
- CI/CD pipeline with GitHub Actions

## Areas for Contribution

### High Priority

- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] CI/CD pipeline
- [ ] Docker containerization
- [ ] Deployment guides (AWS, GCP, Azure)

### Features

- [ ] Player-level statistics integration
- [ ] Injury data incorporation
- [ ] Weather data integration
- [ ] Vegas line comparison
- [ ] ELO rating system
- [ ] Ensemble model combining multiple approaches
- [ ] Live in-game win probability updates

### Improvements

- [ ] Model hyperparameter tuning
- [ ] Additional feature engineering
- [ ] Better error handling
- [ ] Logging improvements
- [ ] Performance optimization
- [ ] Data validation
- [ ] Caching for API responses

### Documentation

- [ ] API documentation (Swagger/OpenAPI)
- [ ] Video tutorials
- [ ] Deployment guides
- [ ] Architecture diagrams
- [ ] Contributing guide enhancements

## Pull Request Guidelines

### PR Title

Use a clear, descriptive title:
- ✅ "Add player injury data integration"
- ✅ "Fix: Correct rolling average calculation"
- ✅ "Docs: Update API endpoint documentation"
- ❌ "Update code"
- ❌ "Fix bug"

### PR Description

Include:
1. **What**: What does this PR do?
2. **Why**: Why is this change needed?
3. **How**: How does it work?
4. **Testing**: How was it tested?
5. **Screenshots**: If UI changes, include screenshots

### Example PR Description

```markdown
## What
Adds integration with player injury data from NFL API

## Why
Injury data is a crucial factor in game outcomes that was previously missing

## How
- Created new data fetcher for injury API
- Added injury features to feature engineering pipeline
- Updated model to include injury-related features

## Testing
- Tested with 2023-2024 season data
- Verified feature calculation accuracy
- Model accuracy improved from 65% to 67%

## Breaking Changes
None - backward compatible
```

## Code Review Process

1. **Automated Checks**: Must pass linting and tests (when implemented)
2. **Peer Review**: At least one maintainer must approve
3. **Documentation**: Ensure docs are updated
4. **Testing**: Verify changes work as expected

## Commit Message Guidelines

Use conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add weather data integration
fix: correct rolling average calculation bug
docs: update API endpoint documentation
refactor: improve feature engineering performance
```

## Questions or Issues?

- Open an issue on GitHub
- Tag maintainers for help
- Check existing issues and PRs first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort! 🙏

