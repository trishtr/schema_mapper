# Schema Mapper

An intelligent schema mapping system that uses embedding models, vector search, and machine learning to automatically map schemas between multiple source databases and a target database.

## Features

-  Intelligent schema mapping using embedding models
-  Vector similarity search for finding related columns
-  LLM integration for complex mapping scenarios
-  Multiple source database support
-  High-performance caching and fallback mechanisms
-  Asynchronous processing
-  Robust error handling

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp docs/config.env.example .env
# Edit .env with your settings
```

3. Run the application:

```bash
python -m src.app.main
```

## Documentation

See the `docs/` directory for detailed documentation.
